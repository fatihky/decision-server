"""Backend for PEFT LoRA decision models that score option letters.

A large group of decision models ship as a PEFT adapter over a base model plus
a small ``head.pt`` decision head. Inference scores one sequence per option
(using A/B/C... letter markers) and softmaxes the head scores.

Configuration keys:

- ``repo_id`` (required): repository with ``adapter_config.json`` and ``head.pt``.
- ``base_model``: base checkpoint (defaults to the adapter's ``base_model_name_or_path``).
- ``revision`` / ``base_revision``: revisions for each.
- ``head_file``: head weights filename (default ``head.pt``).
- ``adapter_subfolder``: subdirectory holding the adapter (default root).
- ``device``, ``dtype``, ``temperature``, ``lm_scale``, ``context_limit``.

This is a faithful re-implementation of the published scoring path; it does not
import family code. For families whose head differs, use ``repo-inference``.

The backbone class follows the adapter's ``task_type``: ``CAUSAL_LM`` loads
``AutoModelForCausalLM``, anything else (feature-extraction adapters such as kev
and lev) loads the bare ``AutoModel`` so the LoRA tensors line up.
"""

from __future__ import annotations

import importlib
import json
import warnings
from typing import Any

from ..backend import BackendError
from ..schema import DecisionRequest, Prediction, Usage, check_probabilities
from .code_loader import snapshot_code_files

LETTERS = [chr(ord("A") + i) for i in range(26)]


class PeftDecisionBackend:
    def __init__(self, config: dict):
        repo_id = config.get("repo_id")
        if not repo_id:
            raise BackendError("peft-decision requires a 'repo_id'")
        self.repo_id = repo_id
        self.revision_arg = config.get("revision")
        self.base_revision = config.get("base_revision")
        self.base_model = config.get("base_model")
        self.head_file = config.get("head_file", "head.pt")
        self.adapter_subfolder = config.get("adapter_subfolder")
        self.device = config.get("device", "cpu")
        self.dtype = config.get("dtype", "float32")
        self.temperature = float(config.get("temperature", 1.0))
        self.lm_scale = float(config.get("lm_scale", 1.0))
        self.context_limit = int(config.get("context_limit", 4096))
        self._usage = Usage()
        self._loaded = False
        self.torch: Any = None
        self.tokenizer: Any = None
        self.model: Any = None
        self.head: Any = None
        self.head_kind: str | None = None
        self.head_meta: dict = {}
        self._feature_extraction = False

    def _load(self) -> None:
        if self._loaded:
            return
        try:
            self.torch = importlib.import_module("torch")
            transformers = importlib.import_module("transformers")
            peft = importlib.import_module("peft")
        except ImportError as exc:
            raise BackendError("Install the open-models extra (torch, transformers, peft)") from exc

        directory = snapshot_code_files(
            self.repo_id,
            self.revision_arg,
            [f for f in [self._adapter_config_name(), self.head_file] if f],
        )
        adapter_config = json.loads((directory / self._adapter_config_name()).read_text())
        base = self.base_model or adapter_config.get("base_model_name_or_path")
        if not base:
            raise BackendError("No base model configured and none in the adapter config")

        dtype = self.torch.bfloat16 if self.dtype == "bfloat16" else self.torch.float32
        tokenizer_kwargs: dict[str, Any] = {}
        if self.revision_arg:
            tokenizer_kwargs["revision"] = self.revision_arg
        if self.adapter_subfolder:
            tokenizer_kwargs["subfolder"] = self.adapter_subfolder
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.repo_id, **tokenizer_kwargs
        )
        backbone_kwargs: dict[str, Any] = {"dtype": dtype}
        if self.base_revision:
            backbone_kwargs["revision"] = self.base_revision
        # Adapters trained for feature extraction wrap the bare backbone, so their
        # tensors are keyed ``base_model.model.layers...``. A causal-LM wrapper adds
        # another ``model.`` level and PEFT would silently drop every adapter tensor.
        task_type = str(adapter_config.get("task_type") or "").upper()
        self._feature_extraction = task_type not in ("", "CAUSAL_LM")
        auto_class = (
            transformers.AutoModel
            if self._feature_extraction
            else transformers.AutoModelForCausalLM
        )
        backbone = auto_class.from_pretrained(base, **backbone_kwargs)
        peft_kwargs = {"subfolder": self.adapter_subfolder} if self.adapter_subfolder else {}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = peft.PeftModel.from_pretrained(backbone, self.repo_id, **peft_kwargs)
        for record in caught:
            message = str(record.message)
            if "missing adapter keys" in message:
                raise BackendError(
                    f"PEFT did not apply the adapter weights from {self.repo_id}: {message[:200]}"
                )
            warnings.warn(message, record.category, stacklevel=1)
        self.model = model.to(self.device).eval().requires_grad_(False)
        self.head = self._load_head(directory / self.head_file)
        self._loaded = True

    def _adapter_config_name(self) -> str:
        if self.adapter_subfolder:
            return f"{self.adapter_subfolder}/adapter_config.json"
        return "adapter_config.json"

    def _load_head(self, path):
        state = self.torch.load(path, map_location=self.device, weights_only=False)
        if isinstance(state, dict):
            self.head_meta = {k: v for k, v in state.items() if k != "head"}
        else:
            self.head_meta = {}
        head_state = state.get("head", state) if isinstance(state, dict) else state
        if isinstance(head_state, dict) and "q.weight" in head_state:
            self.head_kind = "qk"
            self.head = self.torch.nn.Module()
            self.head.q = self.torch.nn.Linear(
                head_state["q.weight"].shape[1], head_state["q.weight"].shape[0]
            )
            self.head.k = self.torch.nn.Linear(
                head_state["k.weight"].shape[1], head_state["k.weight"].shape[0]
            )
            load = self.head.load_state_dict(
                {"q.weight": head_state["q.weight"], "q.bias": head_state["q.bias"],
                 "k.weight": head_state["k.weight"], "k.bias": head_state["k.bias"]}
            )
            _ = load
            return self.head.to(self.device).float().eval().requires_grad_(False)
        # Fallback: a plain scalar head.
        self.head_kind = "linear"
        if isinstance(head_state, dict) and "weight" in head_state:
            weight, bias = head_state["weight"], head_state.get("bias")
        elif isinstance(head_state, dict) and "state_dict" in head_state:
            weight = head_state["state_dict"]["weight"]
            bias = head_state["state_dict"].get("bias")
        else:
            weight, bias = head_state, None
        weight = weight.to(self.device).float()
        module = self.torch.nn.Linear(weight.shape[1], weight.shape[0])
        with self.torch.no_grad():
            module.weight.copy_(weight)
            if bias is not None:
                module.bias.copy_(bias.to(self.device).float())
        return module.to(self.device).eval().requires_grad_(False)

    def _encode(self, text: str):
        encoded = self.tokenizer(text, return_tensors="pt", truncation=False)
        if encoded["input_ids"].shape[1] > self.context_limit:
            raise BackendError(
                f"Input requires {encoded['input_ids'].shape[1]} tokens, "
                f"exceeding {self.context_limit}"
            )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with self.torch.no_grad():
            if self._feature_extraction:
                output = self.model(**encoded)
                return output.last_hidden_state[:, -1, :]
            output = self.model(**encoded, output_hidden_states=True)
        return output.hidden_states[-1][:, -1, :]

    def _score_option(
        self, state_text: str, question_text: str, option_text: str, letter: str
    ) -> float:
        context = (
            f"State:\n{state_text}\n\nQuestion:\n{question_text}\n\n"
            f"Options:\n{letter}. {option_text}\n\nAnswer: {letter}"
        )
        hidden = self._encode(context)
        with self.torch.no_grad():
            if self.head_kind == "qk":
                query = self.head.q(hidden)
                key = self.head.k(self._encode(f"{option_text} {letter}"))
                return float((query * key).sum(dim=-1)[0].detach())
            return float(self.head(hidden)[0, 0].detach())

    def predict(self, request: DecisionRequest) -> list[Prediction]:
        self._load()
        state_text = (
            request.state
            if isinstance(request.state, str)
            else json.dumps(request.state, ensure_ascii=False, indent=2)
        )
        predictions: list[Prediction] = []
        for q in request.questions:
            options = _ordered_options(q)
            if len(options) > len(LETTERS):
                predictions.append(Prediction(question_id=q.id, error="Too many options"))
                continue
            scores = [
                self._score_option(state_text, q.instructions, o.description, LETTERS[i])
                for i, o in enumerate(options)
            ]
            probs = self._softmax(scores)
            result = {o.id: p for o, p in zip(options, probs, strict=True)}
            check_probabilities(result, [o.id for o in q.options])
            predictions.append(Prediction(question_id=q.id, probabilities=result))
            self._usage.requests += 1
        return predictions

    def _softmax(self, scores: list[float]) -> list[float]:
        torch = self.torch
        tensor = torch.tensor(scores, dtype=torch.float32) / max(self.temperature, 1e-6)
        probs = torch.softmax(tensor, dim=0)
        return [float(p) for p in probs]

    def revision(self) -> str | None:
        return self.revision_arg or self.repo_id

    def settings(self) -> dict:
        return {
            "repo_id": self.repo_id,
            "base_model": self.base_model,
            "head_file": self.head_file,
            "temperature": self.temperature,
            "transport": "peft-adapter-lora-head",
        }

    def usage(self) -> Usage:
        return self._usage.model_copy()

    def close(self) -> None:
        self._loaded = False
        self.model = None
        self.head = None


def _ordered_options(question):
    """Score options in declared order; map low-to-high for score scales."""

    if question.task == "score":
        return sorted(question.options, key=lambda o: o.value or 0.0)
    return list(question.options)


def build(config: dict) -> PeftDecisionBackend:
    return PeftDecisionBackend(config)
