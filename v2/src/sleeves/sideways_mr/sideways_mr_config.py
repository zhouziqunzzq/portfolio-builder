# sideways_mr_config.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class SidewaysMRConfig:
    # Universe
    # tickers: List[str] = field(
    #     default_factory=lambda: [
    #         # --- ETFs (core) ---
    #         "XLB",
    #         "XLC",
    #         "XLE",
    #         "XLF",
    #         "XLI",
    #         "XLK",
    #         "XLP",
    #         "XLRE",
    #         "XLU",
    #         "XLV",
    #         "XLY",
    #         "IWM",
    #         "QQQ",
    #         "DIA",
    #     ]
    # )
    tickers: List[str] = field(
        default_factory=lambda: [
            # --- Tech / Communication ---
            "AAPL",
            "MSFT",
            "GOOGL",
            "AMZN",
            "META",
            "NVDA",
            "AVGO",
            "ORCL",
            "CRM",
            "AMD",
            "QCOM",
            # --- Financials ---
            "JPM",
            "BAC",
            "WFC",
            "GS",
            "MS",
            "SCHW",
            "BLK",
            "V",
            # --- Healthcare ---
            "UNH",
            "JNJ",
            "ABBV",
            "MRK",
            "TMO",
            "ABT",
            # --- Consumer (Staples + Discretionary) ---
            "PG",
            "KO",
            "PEP",
            "WMT",
            "COST",
            "HD",
            "LOW",
            "MCD",
            # --- Industrials / Energy ---
            "CAT",
            "DE",
            "UPS",
            "XOM",
            "CVX",
            # --- Utilities / REIT (MR-friendly, optional but recommended) ---
            "NEE",
            "DUK",
            "AMT",
            "PLD",
        ]
    )

    # Spread definition: ticker vs benchmark
    benchmark: str = "SPY"  # default benchmark if not specified in mapping
    benchmark_by_ticker: Dict[str, str] = field(
        default_factory=lambda: {
            # --- Tech ---
            "AAPL": "XLK",
            "MSFT": "XLK",
            "NVDA": "XLK",
            "AVGO": "XLK",
            "ORCL": "XLK",
            "CRM": "XLK",
            "AMD": "XLK",
            "QCOM": "XLK",
            # --- Communication Services ---
            "GOOGL": "XLC",
            "META": "XLC",
            # --- Consumer Discretionary ---
            "AMZN": "XLY",
            "HD": "XLY",
            "LOW": "XLY",
            "MCD": "XLY",
            # --- Consumer Staples ---
            "PG": "XLP",
            "KO": "XLP",
            "PEP": "XLP",
            "WMT": "XLP",
            "COST": "XLP",
            # --- Financials ---
            "JPM": "XLF",
            "BAC": "XLF",
            "WFC": "XLF",
            "GS": "XLF",
            "MS": "XLF",
            "SCHW": "XLF",
            "BLK": "XLF",
            "V": "XLF",
            # --- Healthcare ---
            "UNH": "XLV",
            "JNJ": "XLV",
            "ABBV": "XLV",
            "MRK": "XLV",
            "TMO": "XLV",
            "ABT": "XLV",
            # --- Industrials ---
            "CAT": "XLI",
            "DE": "XLI",
            "UPS": "XLI",
            # --- Energy ---
            "XOM": "XLE",
            "CVX": "XLE",
            # --- Utilities ---
            "NEE": "XLU",
            "DUK": "XLU",
            # --- Real Estate ---
            "AMT": "XLRE",
            "PLD": "XLRE",
        }
    )

    # Signal windows
    spread_window: int = 20  # z-score window
    gate_window: int = 10  # persistence window
    trend_slope_window: int = 20  # slope window for spread drift
    signals_extra_buffer_days: int = 10

    # Signal options
    use_beta_hedged_spread: bool = True
    hedge_window: int = 60  # window for beta estimation

    # Bandwidth gate (scale-aware)
    use_bw_rank: bool = True
    bw_rank_window: int = 252          # history length used to compute rank
    bw_rank_enter: float = 0.35        # rank must be <= this to count as "tight"
    bw_rank_exit: float = 0.60

    # Slope gate (scale-aware)
    use_slope_t: bool = True
    slope_t_window: int = 20           # window used to estimate slope significance
    slope_t_soft: float = 1.0   # <= soft => score ~1
    slope_t_hard: float = 2.5   # >= hard => score = 0 (and can optionally hard-block)
    # optional: hard block when extreme
    slope_t_hard_block: bool = True

    # weighted gate blending
    gate_w_bw: float = 0.45
    gate_w_slope: float = 0.35
    gate_w_bw_slope: float = 0.20

    # Optional: smooth bw before bw_slope check
    bw_ewm_halflife: int = 5
    bw_slope_lookback: int = 10

    # DEPRECATED: Gate thresholds (applied to spread)
    bw_thresh: float = 0.06  # spread bandwidth threshold
    bw_thresh_hard: float = 0.12
    slope_ann_thresh: float = 0.20  # annualized drift threshold
    slope_ann_thresh_hard: float = 0.50
    bw_slope_max: float = 0.25  # allow non-increasing bandwidth
    bw_slope_hard: float = 0.60

    # Gate hysteresis on persistence score
    gate_enter: float = 0.65
    gate_entry_min: float = 0.58
    gate_exit: float = 0.45
    gate_hold: float = 0.45         # minimum gate_score to allow holding
    gate_hard_off: float = 0.25     # only force-exit if gate_score <= this
    gate_off_confirm_days: int = 2  # require N consecutive days below hard_off
    exit_on_gate_off_hard_only: bool = True

    # MR entry/exit on spread z
    entry_z: float = 1.25
    exit_z: float = 0.20
    exit_on_gate_off: bool = True

    # Position constraints
    max_positions: int = 8
    min_hold_days: int = 1
    max_hold_days: int = 10
    cooldown_days: int = 3  # days to wait after exit before re-entry

    # Sizing
    sleeve_gross_cap: float = 0.30  # sleeve exposure cap (sum of weights)
    w_max_per_asset: float = 0.08  # per-name cap inside sleeve
    use_inverse_vol: bool = True
    vol_window: int = 20

    # Liquidity filters
    min_price: float = 3.0
    min_adv: float = 300_000
    min_adv_window: int = 20

    def get_universe(self, include_benchmarks: bool = False) -> List[str]:
        tickers = set([t.upper() for t in self.tickers])
        if include_benchmarks:
            for t in self.tickers:
                bm = self.benchmark_by_ticker.get(t)
                if bm and bm.upper() not in tickers:
                    tickers.add(bm.upper())
        return sorted(tickers)
