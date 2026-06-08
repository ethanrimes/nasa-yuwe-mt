"""Shared infrastructure for the Nasa Yuwe MT monorepo.

Modules:
    config     — Azure IDs, storage layout, model registry, paths.
    blob       — Azure Blob upload / download / mirror helpers.
    h100       — provision / deprovision an Azure H100 VM (idempotent teardown).
    mirror     — background loop: local checkpoints + metrics -> Blob.
    bible      — fetch real WEB(en) + CUV(zh) text indexed by book/chapter/verse.
    translate  — Gemini-subagent batch translation driver (agent-fulfilled queue).
"""

from __future__ import annotations

__all__ = ["config", "blob", "h100", "mirror", "bible", "translate"]
__version__ = "0.1.0"
