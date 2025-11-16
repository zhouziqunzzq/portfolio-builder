from __future__ import annotations

"""
Thin production wrapper for existing MarketDataStore to avoid duplication.
It adjusts sys.path so we can import the project-level implementation.
"""

from pathlib import Path
import sys

# Add project root to sys.path (production/src/ -> project root is 2 levels up)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from market_data_store import MarketDataStore as _MarketDataStore
except Exception as e:
    raise ImportError(f"Failed to import MarketDataStore from project root: {e}")

# Re-export under the same name for production modules to use.
MarketDataStore = _MarketDataStore
