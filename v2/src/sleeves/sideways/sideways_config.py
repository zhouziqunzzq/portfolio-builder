from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class SidewaysConfig:
    # -----------------------------
    # Universe
    # -----------------------------
    tickers: List[str] = field(
        default_factory=lambda: [
            # --- ETFs (core) ---
            "XLU",
            "XLP",
            "XLV",
            "VNQ",
            "USMV",
            "SPLV",
            # --- Defensive stocks ---
            "JNJ",
            "PG",
            "KO",
            "PEP",
            "WMT",
            "MCD",
            # --- Low-beta megacaps ---
            "MSFT",
            "AAPL",
            "CSCO",
            "ORCL",
            "VZ",
            "T",
        ]
    )

    # Liquidity filters (optional; same spirit as DefensiveSleeve)
    min_price: float = 5.0
    min_adv: float = 1e6  # shares/day (simple proxy)
    min_adv_window: int = 20

    # -----------------------------
    # Sideways gate (BB bandwidth + trend slope)
    # -----------------------------
    bb_window: int = 20
    bb_k: float = 2.0
    bw_thresh: float = 0.045
    bw_slope_window: int = 5
    bw_slope_max: float = 0.02
    trend_slope_window: int = 63
    slope_ann_thresh: float = 0.06  # annualized slope threshold

    gate_window: int = 8  # persistence window
    gate_enter: float = 0.6
    gate_exit: float = 0.4

    signals_extra_buffer_days: int = 40  # conservative buffer for rolling windows

    # -----------------------------
    # Mean-reversion trading rule
    # -----------------------------
    entry_z: float = 1.25  # buy if bb_z <= -entry_z
    # For MVP we don't explicitly "exit" inside sleeve weights; we just assign
    # weight to candidates. But you can use exit_z in a stateful version.
    exit_z: float = 0.25  # consider "done" when bb_z >= -exit_z
    exit_on_gate_off: bool = False  # exit positions when no longer sideways

    # -----------------------------
    # Weighting
    # -----------------------------
    use_inverse_vol: bool = True
    vol_window: int = 20
    w_max_per_asset: float = 0.33  # cap within sleeve
    max_positions: int = 3  # max number of simultaneous positions
