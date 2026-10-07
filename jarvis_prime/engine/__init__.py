"""J-Prime Engine -- J-Prime serving GGUF models on a local or cloud GPU.

The Mind of the Trinity serves its own models. This package supervises
llama.cpp's ``llama-server`` (one process per resident model) and exposes the
protocol O+V's local lane already speaks: the OpenAI surface for generation
plus the model-management surface (``/api/tags``, ``/api/ps``, ``/api/show``,
``/api/chat``, ``/api/generate``) that O+V probes for residency, context
physics, eviction and preflight. Nothing in JARVIS has to change to move from
an Ollama daemon to J-Prime: the same twelve calls get the same answers.

Run it with ``python -m jarvis_prime.engine`` (see ``__main__``).

Import-light by design: no torch, no llama-cpp-python, no AGI hub. The engine
binary does the inference; this package owns lifecycle, admission and
protocol.
"""
from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
