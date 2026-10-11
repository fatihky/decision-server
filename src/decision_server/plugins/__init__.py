"""Reusable backend plugins for decision-model families.

Each module exposes ``build(config) -> Backend`` and is referenced from the
catalog. The three plugins here cover the mechanical patterns used across the
Hub ecosystem:

- :mod:`repo_inference` — use the ``inference.py`` / ``model.py`` a repository
  ships and call its ``decide``/``predict`` entry point.
- :mod:`peft_decision` — a PEFT LoRA adapter plus a decision head, scored over
  the option letters.
- :mod:`reranker` — a cross-encoder that scores each option against the state.

Family-specific behaviour is expressed through configuration, not new code.
"""
