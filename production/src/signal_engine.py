from __future__ import annotations

# Thin wrapper to reuse existing implementation while allowing future refactor.
from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from signal_engine import SignalEngine as _SignalEngine  # type: ignore
except Exception as e:
    raise ImportError(f"Failed to import SignalEngine from project root: {e}")

SignalEngine = _SignalEngine
