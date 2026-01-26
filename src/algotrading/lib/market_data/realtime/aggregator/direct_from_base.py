from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional, Set, Tuple

from algotrading.lib.types.market_data import Timeframe, BarKey, OHLCVBar
from algotrading.lib.types.trading import InstrumentRef
from algotrading.lib.eventing.md_events import (
    MDBarSubscribeRequest,
    MDBarUnsubscribeRequest,
    BaseBarUpserted,
    BarCompleted,
    BarUpdated,
    BarClosed,
)
from algotrading.lib.calendar.session import BaseSessionCalendar
from algotrading.lib.market_data.bucketing import floor_time, shift_timeframe
from .base import (
    BaseMarketDataAggregatorConfig,
    BaseMarketDataAggregator,
)

# -----------------------------
# Helpers
# -----------------------------


def _is_integer_multiple(outer: Timeframe, inner: Timeframe) -> bool:
    o = outer.seconds
    i = inner.seconds
    return i > 0 and o % i == 0


# -----------------------------
# Internal aggregation state
# -----------------------------


@dataclass
class _BucketAgg:
    """
    Holds base-bar contributions for ONE (ref, tf, bucket_start).
    Keyed by base bar start_ts to support upsert (corrections).
    """

    key: BarKey
    base_tf: Timeframe
    base_bars: Dict[datetime, OHLCVBar] = field(default_factory=dict)
    # Internal states
    _closed: bool = False
    _closed_ts: Optional[datetime] = None  # timestamp when closed
    _expected_base_bar_ts: Set[datetime] = field(default_factory=set)
    _last_updated_ts: Optional[datetime] = None  # updated on each upsert

    def __post_init__(self) -> None:
        # Pre-compute expected base bar start_ts for completeness checking
        n_bars = self.key.tf.seconds // self.base_tf.seconds
        for i in range(n_bars):
            ts = shift_timeframe(self.key.start_ts, self.base_tf, i)
            self._expected_base_bar_ts.add(ts)

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def closed_ts(self) -> Optional[datetime]:
        return self._closed_ts

    def close(self, *, now: Optional[datetime] = None) -> None:
        self._closed = True
        self._closed_ts = now if now is not None else datetime.now(timezone.utc)

    @property
    def completed(self) -> bool:
        """
        Whether this bucket has received all expected base bars.
        """
        return self._expected_base_bar_ts.issubset(self.base_bars)

    def upsert(
        self, base_start_ts: datetime, bar: OHLCVBar, *, now: Optional[datetime] = None
    ) -> Optional[OHLCVBar]:
        prev = self.base_bars.get(base_start_ts)
        self.base_bars[base_start_ts] = bar
        self._last_updated_ts = now if now is not None else datetime.now(timezone.utc)
        return prev

    def compute_ohlcv(
        self,
        *,
        start_ts_override: Optional[datetime] = None,
        end_ts_override: Optional[datetime] = None,
    ) -> Optional[OHLCVBar]:
        """
        Return an aggregated OHLCVBar compatible with md_events BarClosed/BarUpdated payloads.
        """
        if not self.base_bars:
            return None

        items = sorted(self.base_bars.items(), key=lambda kv: kv[0])
        first = items[0][1]
        last = items[-1][1]

        o = first.o
        c = last.c
        h = max(b.h for _, b in items)
        l = min(b.l for _, b in items)
        v = sum(b.v for _, b in items)
        start_ts = (
            start_ts_override if start_ts_override is not None else self.key.start_ts
        )
        end_ts = (
            end_ts_override
            if end_ts_override is not None
            else shift_timeframe(self.key.start_ts, self.key.tf, 1)
        )

        return OHLCVBar(start_ts=start_ts, end_ts=end_ts, o=o, h=h, l=l, c=c, v=v)


# -----------------------------
# Concrete implementation
# -----------------------------


@dataclass(frozen=True)
class DirectFromBaseAggregatorConfig(BaseMarketDataAggregatorConfig):
    """
    Config for DirectFromBaseAggregator.
    Inherits source_name and base_timeframe from BaseMarketDataAggregatorConfig.
    """

    aggregator_name: str = "DirectFromBaseMarketDataAggregator"

    # Finalization policy
    settle_seconds: float = 90.0  # wait after bucket end before closing
    # GC policy
    keep_closed_buckets_seconds: float = (
        3600.0  # keep closed aggs around for 1h (optional)
    )
    keep_base_bars_seconds: float = 7200.0  # keep raw base bars for 2h (tunable)
    keep_derived_bars_seconds: float = 7200.0  # keep derived bars for 2h (tunable)
    evict_incomplete_buckets_seconds: float = (
        3600.0  # evict incomplete aggs after 1h (optional)
    )


class DirectFromBaseAggregator(BaseMarketDataAggregator):
    """
    Direct-from-base aggregation:
      - Adapts base upserts into requested timeframes.
      - Derives each requested TF directly from base bars.
      - Allows subscribing to base_tf for pass-through.
    """

    def __init__(
        self,
        config: DirectFromBaseAggregatorConfig,
        session_calendar: Optional[BaseSessionCalendar] = None,
    ) -> None:
        super().__init__(config)

        self._config = config
        self._source_name = config.source_name
        self._base_tf = config.base_timeframe
        self._session_calendar = session_calendar

        # ref -> set of requested timeframes (INCLUDING base_tf if subscribed for pass-through)
        self._subs: Dict[InstrumentRef, Set[Timeframe]] = {}
        # base store: (ref, base_start_ts) -> base bar
        self._base_store: Dict[Tuple[InstrumentRef, datetime], OHLCVBar] = {}
        # derived bucket aggs: BarKey(ref, tf, start_ts) -> _BucketAgg
        self._bucket_aggs: Dict[BarKey, _BucketAgg] = {}
        # derived store: BarKey(ref, tf, start_ts) -> aggregated bar
        self._derived_store: Dict[BarKey, OHLCVBar] = {}

    @property
    def source_name(self) -> str:
        return self._source_name

    @property
    def base_timeframe(self) -> Timeframe:
        return self._base_tf

    # ----- subscription management -----

    def on_subscribe(self, msg: MDBarSubscribeRequest) -> None:
        for ref in msg.refs:
            s = self._subs.setdefault(ref, set())
            for tf in msg.timeframes:
                if not _is_integer_multiple(tf, self._base_tf):
                    raise ValueError(
                        f"Requested timeframe {tf} must be an integer multiple of base timeframe {self._base_tf}"
                    )
                s.add(tf)

    def on_unsubscribe(self, msg: MDBarUnsubscribeRequest) -> None:
        for ref in msg.refs:
            s = self._subs.get(ref)
            if not s:
                continue
            for tf in msg.timeframes:
                s.discard(tf)
            if not s:
                self._subs.pop(ref, None)

    def subscribed_timeframes(self, ref: InstrumentRef) -> tuple[Timeframe, ...]:
        s = self._subs.get(ref)
        if not s:
            return ()
        return tuple(sorted(s, key=lambda tf: tf.seconds))

    # ----- ingestion + derivation -----

    def on_base_upsert(
        self, ev: BaseBarUpserted, *, now: Optional[datetime] = None
    ) -> Iterable[BarCompleted | BarUpdated | BarClosed]:
        if ev.key.tf != self._base_tf:
            raise ValueError(
                f"BaseBarUpserted tf={ev.key.tf} does not match aggregator base_timeframe={self._base_tf}"
            )

        now = now if now is not None else datetime.now(timezone.utc)
        ref = ev.key.ref
        base_start = ev.key.start_ts
        curr_bar = ev.curr
        # Enforce tz-aware for base_start
        if base_start.tzinfo is None:
            raise ValueError("Base bar start_ts must be timezone-aware")

        # Upsert into base store
        prev_base = self._base_store.get((ref, base_start))
        self._base_store[(ref, base_start)] = curr_bar
        if_corr_base = prev_base is not None and prev_base != curr_bar

        # Generate output events
        outs: List[BarCompleted | BarUpdated | BarClosed] = []
        tfs = self._subs.get(ref)
        # Case 1: No subscribers for this ref -> no output
        if not tfs:
            return outs
        # Case 2: base_tf subscribed for pass-through
        if self._base_tf in tfs:
            base_key = BarKey(ref=ref, tf=self._base_tf, start_ts=base_start)
            if prev_base is None:  # New bar
                outs.append(
                    BarCompleted(
                        ts=now.timestamp(),
                        source=self._config.aggregator_name,
                        key=base_key,
                        bar=curr_bar,
                    )
                )
            elif if_corr_base:  # Correction
                outs.append(
                    BarUpdated(
                        ts=now.timestamp(),
                        source=self._config.aggregator_name,
                        key=base_key,
                        bar=curr_bar,
                        prev=prev_base,
                    )
                )
        # Case 3: Derived tfs
        for tf in tfs:
            if tf == self._base_tf:
                continue  # already handled pass-through above

            # Locate / create bucket agg
            bucket_start = floor_time(base_start, tf, calendar=self._session_calendar)
            bucket_key = BarKey(ref, tf, bucket_start)
            # Upsert base bar into bucket agg
            agg = self._bucket_aggs.setdefault(
                bucket_key, _BucketAgg(bucket_key, self._base_tf)
            )
            agg.upsert(base_start, curr_bar, now=now)

            # Compute aggregated OHLCV
            # Note: This can be incomplete if the bucket is missing base bars; We handle that below.
            computed = agg.compute_ohlcv()
            if computed is None:
                continue

            # Check previous derived bar in store
            prev_derived = self._derived_store.get(bucket_key)
            is_corr_derived = prev_derived is not None and prev_derived != computed

            # Case 3.1: If the new base bar completes the bucket, close the bucket and emit BarCompleted
            if prev_derived is None and agg.completed:
                self._derived_store[bucket_key] = computed
                outs.append(
                    BarCompleted(
                        ts=now.timestamp(),
                        source=self._config.aggregator_name,
                        key=bucket_key,
                        bar=computed,
                    )
                )
            # Case 3.2: If the new base bar corrects an existing derived bar, emit BarUpdated
            elif is_corr_derived:
                self._derived_store[bucket_key] = computed
                outs.append(
                    BarUpdated(
                        ts=now.timestamp(),
                        source=self._config.aggregator_name,
                        key=bucket_key,
                        bar=computed,
                        prev=prev_derived,
                    )
                )
            # Otherwise, either the bucket is incomplete, or the derived bar is unchanged
            # -> no output

        return outs

    def run_gc(self, now: Optional[datetime] = None) -> Iterable[BarClosed]:
        """
        Garbage collection / finalization pass.

        Responsibilities:
        1) Close completed buckets that are past settle window.
        2) Remove old closed bucket state to free memory.
        3) Evict incomplete buckets after a timeout.
        4) Purge old base bars once no longer needed.
        5) Purge old derived bars once no longer needed.

        Returns BarClosed events for buckets finalized in this run.
        """
        now_dt = now or datetime.now(timezone.utc)
        now_ts = now_dt.timestamp()
        settle = self._config.settle_seconds
        keep_closed = self._config.keep_closed_buckets_seconds
        keep_base = self._config.keep_base_bars_seconds
        keep_derived = self._config.keep_derived_bars_seconds
        evict_incomplete = self._config.evict_incomplete_buckets_seconds
        outs: List[BarClosed] = []

        # 1. Close eligible buckets
        # Close rule: bucket is completed, not closed, AND bucket_end + settle < now
        # NOTE: This loop is O(#buckets). Fine for MVP; later you can index by time.
        for key, agg in list(self._bucket_aggs.items()):
            if agg._closed:
                continue
            if not agg.completed:
                continue

            bucket_start = key.start_ts
            bucket_end = shift_timeframe(bucket_start, key.tf, 1)
            if bucket_end.timestamp() + settle > now_ts:
                continue  # still inside settle window

            # Finalize
            agg.close(now=now_dt)
            computed = agg.compute_ohlcv()
            if not computed:
                continue
            self._derived_store[key] = computed
            outs.append(
                BarClosed(
                    ts=now_ts,
                    source=self._config.aggregator_name,
                    key=key,
                    bar=computed,
                )
            )

        # 2. Evict old closed bucket state
        if keep_closed > 0:
            cutoff = now_ts - keep_closed
            for key, agg in list(self._bucket_aggs.items()):
                if not agg._closed:
                    continue
                closed_ts = agg.closed_ts
                if closed_ts is None:
                    continue
                if closed_ts.timestamp() < cutoff:
                    del self._bucket_aggs[key]

        # 3. Evict incomplete buckets after timeout
        if evict_incomplete > 0:
            cutoff = now_ts - evict_incomplete
            for key, agg in list(self._bucket_aggs.items()):
                if agg._closed:
                    continue
                last_updated_ts = agg._last_updated_ts
                if last_updated_ts is not None and last_updated_ts.timestamp() < cutoff:
                    del self._bucket_aggs[key]

        # 4. Purge old base bars
        # Safe purge: keep at least keep_base seconds of base bars by their start timestamps.
        # This is conservative and simple.
        if keep_base > 0:
            cutoff_dt = now_dt - timedelta(seconds=keep_base)
            for (ref, base_start), _ in list(self._base_store.items()):
                if base_start < cutoff_dt:
                    del self._base_store[(ref, base_start)]

        # 5. Purge old derived bars
        # Safe purge: keep at least keep_derived seconds of derived bars by their start timestamps.
        # This is conservative and simple.
        if keep_derived > 0:
            cutoff_dt = now_dt - timedelta(seconds=keep_derived)
            for key, _ in list(self._derived_store.items()):
                if key.start_ts < cutoff_dt:
                    del self._derived_store[key]

        return outs
