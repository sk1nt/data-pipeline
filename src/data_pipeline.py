"""Importable facade for the CLI ``data-pipeline.py`` entrypoint.

Providing this wrapper keeps FastAPI tests and tooling from importing the
hyphenated filename directly while still sharing the exact same app object
and helper functions.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_ENTRYPOINT = _PROJECT_ROOT / "data-pipeline.py"

_spec = importlib.util.spec_from_file_location("data_pipeline_entry", _ENTRYPOINT)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Unable to load data-pipeline entrypoint at {_ENTRYPOINT}")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

app = _module.app
service_manager = _module.service_manager
_trigger_queue_processing = _module._trigger_queue_processing
gex_history_queue = _module.gex_history_queue
