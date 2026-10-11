# decision-server

A local web server and Gradio UI for serving **SystemOne-compatible decision
models** behind one typed contract. It answers three kinds of question:

| Task     | Input                           | Output                          |
| -------- | ------------------------------- | ------------------------------- |
| `choice` | arbitrary state + options       | a probability for every option  |
| `noul`   | arbitrary state + yes/no        | `P(true)` (and `P(false)`)      |
| `score`  | arbitrary state + numeric scale | a probability over each level   |

It can serve **any SystemOne-compatible model on the Hugging Face Hub**:

- **GGUF decision models** are downloaded and served by a managed
  `llama-server`, which implements `/v1/systemone` natively. This is the
  uniform path and covers every model tagged `decision-model` + `gguf`.
- **Transformer decision models** are downloaded and loaded in-process.
- **Cloud models** (TypeSafe, Liquid) are proxied using an API key.
- **Your own models** attach through a plugin.

Weights are downloaded on first use and cached; nothing is bundled.

## Quick start

Requires Python 3.11, [uv](https://docs.astral.sh/uv/), and — for GGUF models —
a `llama-server` on `PATH` (install [llama.cpp](https://github.com/ggml-org/llama.cpp),
e.g. `brew install llama.cpp`) or set `DECISION_SERVER_LLAMA_BIN`.

```sh
uv sync
uv run decision-server --list      # shows every registered model
uv run decision-server             # opens http://127.0.0.1:7860
```

The UI lets you pick a model, load a bundled sample, edit the request JSON and
run it. The first call to a local model downloads its weights.

## What can be served

97 models ship in the bundled catalog:

| Backend | Count | How it runs |
| --- | --- | --- |
| `gguf` | 33 | downloaded, served by `llama-server` |
| `plugin` | 60 | in-process family plugins (see below) |
| `cloud` | 2 | proxied to TypeSafe / Liquid with an API key |
| `automap` | 1 | transformer with `trust_remote_code` |
| `local-http` | 1 | forwarded to a caller-managed server |

```sh
uv run decision-server --list       # id, backend, tasks and availability per model
```

The catalog has three sources:

1. **`models.yaml`** — curated entries: cloud proxies, a caller-managed
   `llama-server`, the offline demo, and transformer `automap` models.
2. **`ecosystem.yaml`** — generated catalog of GGUF decision models on the Hub
   (Laya, Kev, OpenJev, Decider, Clef, TinyJev, Tev, Bosun, Julia, OpenThai,
   StandardOne, NeoHorse, JevK5, APUS, plumb, imajev, spark, hopper, Jeff,
   rune, Jevas, laya-multilingual, …).
3. **`families.yaml`** — repositories with no decision GGUF, served through
   in-process family plugins (see below).

Plus live discovery: browse the Hub `decision-model` tag from the UI or CLI.

One server exposes both interfaces: a JSON HTTP API at the root and the Gradio
UI at `/ui`.

## Family plugins

Repositories that ship no decision GGUF are loaded by one of three plugins,
chosen from their published interface:

| Plugin | Interface it expects | Examples |
| --- | --- | --- |
| `repo-inference` | the repo's own `inference.py`/`model.py` entry point (`DecisionModel.from_pretrained(...).decide(...)`) | mini-Jev, ouro, Manchego, Eikos, Jev-Omni, JevForge, minojev |
| `peft-decision` | a PEFT LoRA adapter plus a `head.pt` decision head, scored over option letters | kev-0.5b/0.6b/8b/27b, lev-350m, ouro, reflex, jebadiah, smalljev |
| `reranker` | a cross-encoder or zero-shot classifier scoring each option against the state | gte-reranker, bge-reranker, mxbai-rerank, zerank, Jeff, von, gliner2 |

Each entry lists the runtime modules it needs under `requires`, so the UI
shows a model as available only once those are installed:

```sh
uv sync --extra open-models          # torch + transformers + peft
# or individually:
uv sync --extra torch --extra transformers --extra peft
```

Family behaviour is configuration, not code: point `repo_id` at the repository
and, where the family returns differently-named keys, add an `option_map`.

`repo-inference` and `reranker` are faithful: they use the family's own
published scoring path. `peft-decision` reconstructs the decision head from the
checkpoint (`head.pt`), which for most such repositories is the only artifact
shipped; its probabilities are directionally correct but not a verified
reproduction of the family scorer, and the catalog says so.

A model is listed as available once its requirements are met: GGUF models need
`llama-server`, transformer models need `torch` + `transformers`, and
`peft-decision` models additionally need `peft`. Cloud models need their API key.

## Request format

```json
{
  "state": { "any": "JSON you want the model to reason over" },
  "questions": [
    {
      "id": "q1",
      "task": "choice",
      "instructions": "Pick the best option.",
      "options": [
        { "id": "a", "description": "First option" },
        { "id": "b", "description": "Second option" }
      ]
    }
  ]
}
```

- `state` may be any JSON value (or a string).
- `instructions` is required and non-empty.
- `choice` needs two or more options with free-form IDs.
- `noul` must use exactly the option IDs `false` and `true`.
- `score` needs a numeric `value` on every option, at least two distinct.
- `system_prompt` and `instructions_json` are optional per question.

Response (one entry per question, in request order):

```json
{
  "model": "ggml-org__Kev-0.8B-GGUF",
  "backend": "gguf",
  "revision": "e551e319d483ff57e1ff208924b349d397cffc1c",
  "settings": {
    "repo_id": "ggml-org/Kev-0.8B-GGUF",
    "gguf_file": "Kev-0.8B-Q8_0.gguf",
    "server": "llama.cpp llama-server",
    "endpoint": "/v1/systemone",
    "managed": "decision-server"
  },
  "usage": { "requests": 1, "input_tokens": 66, "output_tokens": 0 },
  "answers": [
    { "question_id": "urgency", "probabilities": { "routine": 0.71, "priority": 0.24, "urgent": 0.05 } }
  ]
}
```

`revision` is the resolved checkpoint revision, `settings` records the effective
runtime (backend, server, device, temperature), and `usage` accumulates tokens
where the backend reports them. A question that fails carries `"error"` instead
of `probabilities`; the rest of the request still returns.

## HTTP API

Every server (unless started with `--api-only` or `--no-api`) serves a JSON API
alongside the UI:

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | liveness |
| `GET` | `/v1/models` | list models with availability |
| `GET` | `/v1/models/{id}` | one model's description |
| `POST` | `/v1/models` | register a model from the Hub (`{"repo_id": "..."}`) |
| `POST` | `/v1/systemone` | run typed questions against a model |
| `GET` | `/v1/hub/search` | search the Hub `decision-model` tag |
| `POST` | `/v1/hub/describe` | inspect one Hub repository |

`POST /v1/systemone` takes the request body above plus a `model` field:

```sh
curl -X POST http://127.0.0.1:7860/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "ggml-org__Kev-0.8B-GGUF",
    "state": "I was charged twice and nobody replied.",
    "questions": [{
      "id": "route",
      "task": "choice",
      "instructions": "Which team should handle this?",
      "options": [
        {"id": "billing", "description": "payments, charges, refunds"},
        {"id": "shipping", "description": "delivery, tracking"},
        {"id": "technical", "description": "bugs, login"}
      ]
    }]
  }'
```

Errors use standard status codes: `404` for an unknown model, `422` for an
invalid request or unsupported task, `502` for a Hub failure. Interactive docs
are at `/docs`, the OpenAPI schema at `/openapi.json`. The Gradio UI is at
`/ui`.

The same API is available from Python without a server:

```python
from decision_server.api import create_api
from decision_server.registry import default_registry

api = create_api(default_registry())
```

## Browsing the Hub

The UI has a **Search Hugging Face Hub** panel: type a query, filter GGUF-only,
then paste a repo id and press **Add model** to register and serve it live.

The same discovery is available from the CLI and Python:

```sh
uv run decision-server --discover kev --gguf-only
```

```python
from decision_server.hub import search_models, describe

for m in search_models(query="kev", limit=10):
    print(m.id, m.servable_via, m.downloads)   # servable_via: "gguf" | "transformers"

meta = describe("ggml-org/Kev-0.8B-GGUF")
print(meta["is_gguf"], meta["gguf_files"], meta["sha"])
```

To regenerate the bundled GGUF catalog from a repo list:

```sh
python -m decision_server.scripts.generate_catalog repos.txt \
  > src/decision_server/ecosystem.yaml
```

Add `--include-discovered` to also pull recent repositories from the Hub
`decision-model` tag.

## How a model is served

For a given repository the server picks the backend automatically:

1. Tagged `gguf` (or contains `.gguf` files) → **GGUF** via `llama-server`.
2. Ships `modeling_*.py` / `configuration_auto.py` / `custom_code` → **automap**
   (transformer with `trust_remote_code`).
3. Ships `inference.py` / `model.py` / `system_one.py` → **repo-inference**.
4. A PEFT adapter plus `head.pt` → **peft-decision**.
5. A cross-encoder / zero-shot classifier → **reranker**.

Repositories with none of these need a family plugin (see below).

## Cloud models

Set the API key before starting the server; keys are read only from the
environment.

```sh
export TYPESAFE_API_KEY=...
export LIQUID_API_KEY=...
uv run decision-server
```

## Writing a plugin

To serve a family with no GGUF and no standard interface, add a plugin module
and reference it from a catalog file. This is the same mechanism the bundled
`repo-inference`, `peft-decision` and `reranker` plugins use.

```python
# my_package/my_model.py
from decision_server.schema import DecisionRequest, Prediction, Usage

class MyBackend:
    def predict(self, request: DecisionRequest) -> list[Prediction]: ...
    def revision(self): return None
    def settings(self): return {"device": "cuda:0"}
    def usage(self): return Usage()
    def close(self): ...

def build(config: dict) -> MyBackend:
    return MyBackend(**config)
```

```yaml
- id: my-model
  label: My Model
  kind: local
  backend: plugin
  plugin: my_package.my_model
  requires: [torch, transformers]
  config: {device: cuda:0}
  tasks: [choice, noul, score]
```

## CLI

```
--host                   bind address (default 127.0.0.1)
--port                   bind port (default 7860)
--share                  create a public Gradio share link
--api-only               serve only the JSON API (no UI)
--no-api                 serve only the Gradio UI (no JSON API)
--registry PATH          use an alternative catalog instead of the bundled one
--list                   print registered models and exit
--discover [QUERY]       search the Hub for decision models and exit
--gguf-only              with --discover, list only GGUF-servable models
```

`--list` prints one line per model (`id`, `kind/backend`, `tasks`, availability)
and a summary of how many are available:

```sh
uv run decision-server --list
uv run decision-server --discover laya --gguf-only
uv run decision-server --discover kev
```

## Environment

```
DECISION_SERVER_LLAMA_BIN   path to llama-server (else PATH)
DECISION_SERVER_CACHE       cache directory for llama-server logs
TYPESAFE_API_KEY            TypeSafe cloud key
LIQUID_API_KEY              Liquid cloud key
```

## Runtime extras

Local, in-process models need extra runtime packages. Install only what the
models you use require:

```sh
uv sync --extra open-models            # torch + transformers + peft + accelerate
uv sync --extra torch --extra transformers --extra peft   # the same, individually
```

The catalog records each model's needs under `requires`, and the UI/`--list`
report a model as unavailable until they can be imported. GGUF models need no
Python extras — only `llama-server`.

## Layout

```
src/decision_server/
  schema.py       typed-question contract
  backend.py      backend interface
  cloud.py        HTTP proxy backends (bearer token)
  local.py        in-process plugin + local-server backends
  hub.py          Hugging Face discovery
  llama.py        llama-server lifecycle + GGUF download
  servable.py     GGUF and transformer auto-download backends
  registry.py     catalog loader and backend construction
  models.yaml     curated catalog
  ecosystem.yaml  generated GGUF catalog
  families.yaml   generated in-process plugin catalog
  service.py      request execution
  samples.py      bundled example requests
  app.py          Gradio UI
  api.py          JSON HTTP API (FastAPI)
  plugins/
    uniform.py              offline demo
    transformers_classifier.py  generic HF classifier
    repo_inference.py       the repo's own inference code
    peft_decision.py        PEFT LoRA + decision head
    reranker.py             cross-encoder / zero-shot
    code_loader.py          load code a repo ships
  scripts/        catalog generator
```

## Development

```sh
uv run pytest
uv run ruff check .
uv run ty check
```

The test suite runs offline; it never downloads weights or starts a server.
