import pytest

from algotrading.lib.types.market_data import Timeframe, TimeframeUnit, BarKey
from algotrading.lib.market_data.realtime.aggregator.direct_from_base import (
    _is_integer_multiple,
)
from algotrading.lib.market_data.realtime.aggregator.direct_from_base import _BucketAgg
from algotrading.lib.types.market_data import OHLCVBar
from datetime import datetime, timezone, timedelta
from algotrading.lib.market_data.realtime.aggregator.direct_from_base import (
    DirectFromBaseAggregator,
    DirectFromBaseAggregatorConfig,
)
from algotrading.lib.types.instruments import InstrumentRef
from algotrading.lib.eventing.md_events import (
    BaseBarUpserted,
    BarCompleted,
    BarUpdated,
    BarClosed,
    BarBatchCompleted,
    MDBarSubscribeRequest,
    MDBarUnsubscribeRequest,
    MDBarBatchSubscribeRequest,
)


@pytest.mark.parametrize(
    "outer,inner,expected",
    [
        (Timeframe(1, TimeframeUnit.HOUR), Timeframe(30, TimeframeUnit.MINUTE), True),
        (Timeframe(2, TimeframeUnit.HOUR), Timeframe(30, TimeframeUnit.MINUTE), True),
        (Timeframe(15, TimeframeUnit.MINUTE), Timeframe(1, TimeframeUnit.MINUTE), True),
        (Timeframe(1, TimeframeUnit.MINUTE), Timeframe(2, TimeframeUnit.MINUTE), False),
        (Timeframe(1, TimeframeUnit.DAY), Timeframe(1, TimeframeUnit.HOUR), True),
        (Timeframe(1, TimeframeUnit.HOUR), Timeframe(1, TimeframeUnit.SECOND), True),
        (Timeframe(7, TimeframeUnit.SECOND), Timeframe(3, TimeframeUnit.SECOND), False),
    ],
)
def test_is_integer_multiple(outer: Timeframe, inner: Timeframe, expected: bool):
    assert _is_integer_multiple(outer, inner) is expected


def test_bucketagg_empty_returns_none():
    ref = InstrumentRef("AAPL")
    base_tf = Timeframe(30, TimeframeUnit.SECOND)
    derived_tf = Timeframe(30, TimeframeUnit.SECOND)
    t0 = datetime(2021, 1, 1, 12, 0, tzinfo=timezone.utc)
    key = BarKey(ref=ref, tf=derived_tf, start_ts=t0)
    b = _BucketAgg(key, base_tf)
    assert b.compute_ohlcv() is None
    assert not b.completed


def test_bucketagg_single_and_upsert_replace():
    ref = InstrumentRef("AAPL")
    base_tf = Timeframe(60, TimeframeUnit.SECOND)
    derived_tf = Timeframe(60, TimeframeUnit.SECOND)
    t0 = datetime(2021, 1, 1, 12, 0, tzinfo=timezone.utc)
    key = BarKey(ref=ref, tf=derived_tf, start_ts=t0)
    b = _BucketAgg(key, base_tf)
    bar1 = OHLCVBar(
        start_ts=t0,
        end_ts=t0 + timedelta(seconds=60),
        o=1.0,
        h=1.1,
        l=0.9,
        c=1.05,
        v=10.0,
    )
    prev = b.upsert(t0, bar1)
    assert prev is None
    agg = b.compute_ohlcv()
    assert agg == bar1

    # replace existing bar (in case of correction)
    bar2 = OHLCVBar(
        start_ts=t0,
        end_ts=t0 + timedelta(seconds=60),
        o=1.2,
        h=1.3,
        l=1.1,
        c=1.25,
        v=5.0,
    )
    prev = b.upsert(t0, bar2)
    assert prev == bar1
    agg = b.compute_ohlcv()
    assert agg == bar2


def test_bucketagg_multiple_bars_aggregation_and_ordering():
    ref = InstrumentRef("AAPL")
    base_tf = Timeframe(30, TimeframeUnit.SECOND)
    derived_tf = Timeframe(90, TimeframeUnit.SECOND)
    t0 = datetime(2021, 1, 1, 9, 30, tzinfo=timezone.utc)
    t1 = t0 + timedelta(seconds=30)
    t2 = t0 + timedelta(seconds=60)

    bar_a = OHLCVBar(start_ts=t0, end_ts=t1, o=10.0, h=11.0, l=9.5, c=10.5, v=100)
    bar_b = OHLCVBar(start_ts=t1, end_ts=t2, o=10.5, h=12.0, l=10.0, c=11.5, v=150)
    bar_c = OHLCVBar(
        start_ts=t2,
        end_ts=t2 + timedelta(seconds=30),
        o=11.5,
        h=12.5,
        l=11.0,
        c=12.0,
        v=200,
    )

    key = BarKey(ref=ref, tf=derived_tf, start_ts=t0)
    b = _BucketAgg(key, base_tf)

    # insert out of order to ensure sorting by key
    b.upsert(t1, bar_b)
    b.upsert(t0, bar_a)
    b.upsert(t2, bar_c)

    agg = b.compute_ohlcv()
    assert agg.start_ts == key.start_ts
    assert agg.end_ts == t0 + timedelta(seconds=90)
    assert agg.o == bar_a.o
    assert agg.c == bar_c.c
    assert agg.h == max(bar_a.h, bar_b.h, bar_c.h)
    assert agg.l == min(bar_a.l, bar_b.l, bar_c.l)
    assert agg.v == bar_a.v + bar_b.v + bar_c.v


def test_bucketagg_completed_flag():
    ref = InstrumentRef("AAPL")
    base_tf = Timeframe(30, TimeframeUnit.SECOND)
    derived_tf = Timeframe(90, TimeframeUnit.SECOND)
    t0 = datetime(2021, 1, 1, 9, 30, tzinfo=timezone.utc)

    key = BarKey(ref=ref, tf=derived_tf, start_ts=t0)
    b = _BucketAgg(key, base_tf)

    # initially incomplete
    assert not b.completed

    # upsert first base bar
    bar0 = OHLCVBar(
        start_ts=t0, end_ts=t0 + timedelta(seconds=30), o=1, h=2, l=1, c=1.5, v=10
    )
    b.upsert(t0, bar0)
    assert not b.completed

    # upsert second base bar
    t1 = t0 + timedelta(seconds=30)
    bar1 = OHLCVBar(
        start_ts=t1, end_ts=t1 + timedelta(seconds=30), o=1.5, h=2.5, l=1.4, c=2.0, v=20
    )
    b.upsert(t1, bar1)
    assert not b.completed

    # upsert third base bar -> now completed
    t2 = t0 + timedelta(seconds=60)
    bar2 = OHLCVBar(
        start_ts=t2, end_ts=t2 + timedelta(seconds=30), o=2.0, h=3.0, l=1.9, c=2.5, v=30
    )
    b.upsert(t2, bar2)
    assert b.completed

    # extra upserts do not break completed
    extra = OHLCVBar(
        start_ts=t2 + timedelta(seconds=30),
        end_ts=t2 + timedelta(seconds=60),
        o=2.5,
        h=3.5,
        l=2.4,
        c=3.0,
        v=5,
    )
    b.upsert(t2 + timedelta(seconds=30), extra)
    assert b.completed


def test_direct_from_base_subscribe_unsubscribe_and_queries():
    cfg = DirectFromBaseAggregatorConfig(
        source_name="x", base_timeframe=Timeframe(30, TimeframeUnit.SECOND)
    )
    agg = DirectFromBaseAggregator(cfg)

    ref = InstrumentRef("AAPL")
    other = InstrumentRef("MSFT")
    tf_base = Timeframe(30, TimeframeUnit.SECOND)
    tf_60 = Timeframe(60, TimeframeUnit.SECOND)
    tf_90 = Timeframe(90, TimeframeUnit.SECOND)
    tf_bad = Timeframe(99, TimeframeUnit.SECOND)  # not an integer multiple of 30

    # subscribe valid timeframes for two refs
    agg.on_subscribe(
        MDBarSubscribeRequest(
            ts=datetime.now(timezone.utc).timestamp(),
            source="src",
            instrument_refs=(ref, other),
            timeframes=(tf_base, tf_60, tf_90),
        )
    )
    assert agg.subscribed_timeframes(ref) == (tf_base, tf_60, tf_90)
    assert agg.subscribed_timeframes(other) == (tf_base, tf_60, tf_90)

    # subscribing invalid timeframe raises and leaves prior state intact
    with pytest.raises(ValueError):
        agg.on_subscribe(
            MDBarSubscribeRequest(
                ts=datetime.now(timezone.utc).timestamp(),
                source="src",
                instrument_refs=(ref,),
                timeframes=(tf_bad,),
            )
        )
    assert agg.subscribed_timeframes(ref) == (tf_base, tf_60, tf_90)

    # unsubscribe one timeframe
    agg.on_unsubscribe(
        MDBarUnsubscribeRequest(
            ts=datetime.now(timezone.utc).timestamp(),
            source="src",
            instrument_refs=(ref,),
            timeframes=(tf_base,),
        )
    )
    assert agg.subscribed_timeframes(ref) == (tf_60, tf_90)

    # unsubscribe remaining timeframes removes the ref entry
    agg.on_unsubscribe(
        MDBarUnsubscribeRequest(
            ts=datetime.now(timezone.utc).timestamp(),
            source="src",
            instrument_refs=(ref,),
            timeframes=(tf_60, tf_90),
        )
    )
    assert agg.subscribed_timeframes(ref) == ()
    assert ref not in agg._subs

    # other ref should remain subscribed
    assert agg.subscribed_timeframes(other) == (tf_base, tf_60, tf_90)

    # unsubscribing unknown ref should be a no-op
    unknown = InstrumentRef("NVDA")
    agg.on_unsubscribe(
        MDBarUnsubscribeRequest(
            ts=datetime.now(timezone.utc).timestamp(),
            source="src",
            instrument_refs=(unknown,),
            timeframes=(tf_60,),
        )
    )
    assert agg.subscribed_timeframes(unknown) == ()


def test_on_base_upsert_case1_no_subscribers():
    cfg = DirectFromBaseAggregatorConfig(
        source_name="x", base_timeframe=Timeframe(30, TimeframeUnit.SECOND)
    )
    agg = DirectFromBaseAggregator(cfg)

    ref = InstrumentRef("AAPL")
    base_tf = Timeframe(30, TimeframeUnit.SECOND)
    t0 = datetime(2021, 1, 1, 9, 30, tzinfo=timezone.utc)
    bar = OHLCVBar(
        start_ts=t0, end_ts=t0 + timedelta(seconds=30), o=1, h=2, l=1, c=1.5, v=10
    )
    ev = BaseBarUpserted(
        ts=t0.timestamp(),
        source="src",
        key=BarKey(ref=ref, tf=base_tf, start_ts=t0),
        curr=bar,
        prev=None,
        is_correction=False,
    )

    outs = list(agg.on_base_upsert(ev, now=t0))
    assert outs == []


def test_on_base_upsert_case2_base_pass_through_new_and_correction():
    cfg = DirectFromBaseAggregatorConfig(
        source_name="x", base_timeframe=Timeframe(30, TimeframeUnit.SECOND)
    )
    agg = DirectFromBaseAggregator(cfg)

    ref = InstrumentRef("AAPL")
    base_tf = Timeframe(30, TimeframeUnit.SECOND)
    agg.on_subscribe(
        MDBarSubscribeRequest(
            ts=datetime.now(timezone.utc).timestamp(),
            source="src",
            instrument_refs=(ref,),
            timeframes=(base_tf,),
        )
    )

    t0 = datetime(2021, 1, 1, 9, 30, tzinfo=timezone.utc)
    bar1 = OHLCVBar(
        start_ts=t0, end_ts=t0 + timedelta(seconds=30), o=1, h=2, l=1, c=1.5, v=10
    )
    ev1 = BaseBarUpserted(
        ts=t0.timestamp(),
        source="src",
        key=BarKey(ref=ref, tf=base_tf, start_ts=t0),
        curr=bar1,
        prev=None,
        is_correction=False,
    )

    outs = list(agg.on_base_upsert(ev1, now=t0))
    assert len(outs) == 1
    assert isinstance(outs[0], BarCompleted)
    assert outs[0].key.tf == base_tf
    assert outs[0].bar == bar1

    # correction on same base bar
    bar2 = OHLCVBar(
        start_ts=t0, end_ts=t0 + timedelta(seconds=30), o=1.1, h=2.1, l=1.0, c=1.6, v=12
    )
    ev2 = BaseBarUpserted(
        ts=t0.timestamp(),
        source="src",
        key=BarKey(ref=ref, tf=base_tf, start_ts=t0),
        curr=bar2,
        prev=bar1,
        is_correction=True,
    )
    outs = list(agg.on_base_upsert(ev2, now=t0))
    assert len(outs) == 1
    assert isinstance(outs[0], BarUpdated)
    assert outs[0].key.tf == base_tf
    assert outs[0].bar == bar2
    assert outs[0].prev == bar1


def test_on_base_upsert_case3_derived_aggregation_and_correction():
    cfg = DirectFromBaseAggregatorConfig(
        source_name="x", base_timeframe=Timeframe(30, TimeframeUnit.SECOND)
    )
    agg = DirectFromBaseAggregator(cfg)

    ref = InstrumentRef("AAPL")
    base_tf = Timeframe(30, TimeframeUnit.SECOND)
    derived_tf = Timeframe(90, TimeframeUnit.SECOND)
    agg.on_subscribe(
        MDBarSubscribeRequest(
            ts=datetime.now(timezone.utc).timestamp(),
            source="src",
            instrument_refs=(ref,),
            timeframes=(derived_tf,),
        )
    )

    t0 = datetime(2021, 1, 1, 9, 30, tzinfo=timezone.utc)
    t1 = t0 + timedelta(seconds=30)
    t2 = t0 + timedelta(seconds=60)

    bar0 = OHLCVBar(start_ts=t0, end_ts=t1, o=1, h=2, l=1, c=1.5, v=10)
    bar1 = OHLCVBar(start_ts=t1, end_ts=t2, o=1.5, h=2.5, l=1.2, c=2.0, v=20)
    bar2 = OHLCVBar(
        start_ts=t2, end_ts=t2 + timedelta(seconds=30), o=2.0, h=3.0, l=1.8, c=2.5, v=30
    )

    ev0 = BaseBarUpserted(
        ts=t0.timestamp(),
        source="src",
        key=BarKey(ref=ref, tf=base_tf, start_ts=t0),
        curr=bar0,
        prev=None,
        is_correction=False,
    )
    ev1 = BaseBarUpserted(
        ts=t1.timestamp(),
        source="src",
        key=BarKey(ref=ref, tf=base_tf, start_ts=t1),
        curr=bar1,
        prev=None,
        is_correction=False,
    )
    ev2 = BaseBarUpserted(
        ts=t2.timestamp(),
        source="src",
        key=BarKey(ref=ref, tf=base_tf, start_ts=t2),
        curr=bar2,
        prev=None,
        is_correction=False,
    )

    assert list(agg.on_base_upsert(ev0, now=t0)) == []
    assert list(agg.on_base_upsert(ev1, now=t1)) == []
    outs = list(agg.on_base_upsert(ev2, now=t2))
    assert len(outs) == 1
    assert isinstance(outs[0], BarCompleted)
    assert outs[0].key.tf == derived_tf
    assert outs[0].bar.o == bar0.o
    assert outs[0].bar.c == bar2.c
    assert outs[0].bar.v == bar0.v + bar1.v + bar2.v

    # correction after derived bar exists -> BarUpdated
    bar1_corr = OHLCVBar(start_ts=t1, end_ts=t2, o=1.6, h=2.6, l=1.3, c=2.1, v=25)
    ev1_corr = BaseBarUpserted(
        ts=t1.timestamp(),
        source="src",
        key=BarKey(ref=ref, tf=base_tf, start_ts=t1),
        curr=bar1_corr,
        prev=bar1,
        is_correction=True,
    )
    outs = list(agg.on_base_upsert(ev1_corr, now=t2))
    assert len(outs) == 1
    assert isinstance(outs[0], BarUpdated)
    assert outs[0].key.tf == derived_tf
    assert outs[0].bar.v == bar0.v + bar1_corr.v + bar2.v


def test_run_gc_closes_completed_buckets_after_settle():
    cfg = DirectFromBaseAggregatorConfig(
        source_name="x",
        base_timeframe=Timeframe(30, TimeframeUnit.SECOND),
        settle_seconds=0,
        keep_closed_buckets_seconds=0,
        keep_base_bars_seconds=0,
        keep_derived_bars_seconds=0,
        evict_incomplete_buckets_seconds=0,
    )
    agg = DirectFromBaseAggregator(cfg)

    ref = InstrumentRef("AAPL")
    base_tf = Timeframe(30, TimeframeUnit.SECOND)
    derived_tf = Timeframe(90, TimeframeUnit.SECOND)
    agg.on_subscribe(
        MDBarSubscribeRequest(
            ts=datetime.now(timezone.utc).timestamp(),
            source="src",
            instrument_refs=(ref,),
            timeframes=(derived_tf,),
        )
    )

    t0 = datetime(2021, 1, 1, 9, 30, tzinfo=timezone.utc)
    t1 = t0 + timedelta(seconds=30)
    t2 = t0 + timedelta(seconds=60)
    t3 = t0 + timedelta(seconds=120)

    bar0 = OHLCVBar(start_ts=t0, end_ts=t1, o=1, h=2, l=1, c=1.5, v=10)
    bar1 = OHLCVBar(start_ts=t1, end_ts=t2, o=1.5, h=2.5, l=1.2, c=2.0, v=20)
    bar2 = OHLCVBar(
        start_ts=t2, end_ts=t2 + timedelta(seconds=30), o=2.0, h=3.0, l=1.8, c=2.5, v=30
    )

    ev0 = BaseBarUpserted(
        ts=t0.timestamp(),
        source="src",
        key=BarKey(ref=ref, tf=base_tf, start_ts=t0),
        curr=bar0,
        prev=None,
        is_correction=False,
    )
    ev1 = BaseBarUpserted(
        ts=t1.timestamp(),
        source="src",
        key=BarKey(ref=ref, tf=base_tf, start_ts=t1),
        curr=bar1,
        prev=None,
        is_correction=False,
    )
    ev2 = BaseBarUpserted(
        ts=t2.timestamp(),
        source="src",
        key=BarKey(ref=ref, tf=base_tf, start_ts=t2),
        curr=bar2,
        prev=None,
        is_correction=False,
    )

    assert list(agg.on_base_upsert(ev0, now=t0)) == []
    assert list(agg.on_base_upsert(ev1, now=t1)) == []
    assert len(list(agg.on_base_upsert(ev2, now=t2))) == 1

    outs = list(agg.run_gc(now=t3))
    assert len(outs) == 1
    assert isinstance(outs[0], BarClosed)
    assert outs[0].key.tf == derived_tf
    assert outs[0].bar.v == bar0.v + bar1.v + bar2.v


def test_run_gc_evicts_closed_buckets():
    cfg = DirectFromBaseAggregatorConfig(
        source_name="x",
        base_timeframe=Timeframe(30, TimeframeUnit.SECOND),
        settle_seconds=0,
        keep_closed_buckets_seconds=10,
        keep_base_bars_seconds=0,
        keep_derived_bars_seconds=0,
        evict_incomplete_buckets_seconds=0,
    )
    agg = DirectFromBaseAggregator(cfg)

    ref = InstrumentRef("AAPL")
    base_tf = Timeframe(30, TimeframeUnit.SECOND)
    derived_tf = Timeframe(90, TimeframeUnit.SECOND)
    t0 = datetime(2021, 1, 1, 9, 30, tzinfo=timezone.utc)
    key = BarKey(ref=ref, tf=derived_tf, start_ts=t0)

    bucket = _BucketAgg(key, base_tf)
    bucket.close(now=t0 - timedelta(seconds=20))
    agg._bucket_aggs[key] = bucket

    assert key in agg._bucket_aggs
    list(agg.run_gc(now=t0))
    assert key not in agg._bucket_aggs


def test_run_gc_purges_base_and_derived_stores():
    cfg = DirectFromBaseAggregatorConfig(
        source_name="x",
        base_timeframe=Timeframe(30, TimeframeUnit.SECOND),
        settle_seconds=0,
        keep_closed_buckets_seconds=0,
        keep_base_bars_seconds=10,
        keep_derived_bars_seconds=10,
        evict_incomplete_buckets_seconds=0,
    )
    agg = DirectFromBaseAggregator(cfg)

    ref = InstrumentRef("AAPL")
    derived_tf = Timeframe(90, TimeframeUnit.SECOND)
    now = datetime(2021, 1, 1, 10, 0, tzinfo=timezone.utc)
    old = now - timedelta(seconds=20)

    old_base = OHLCVBar(
        start_ts=old, end_ts=old + timedelta(seconds=30), o=1, h=1, l=1, c=1, v=1
    )
    agg._base_store[(ref, old)] = old_base

    old_key = BarKey(ref=ref, tf=derived_tf, start_ts=old)
    old_derived = OHLCVBar(
        start_ts=old, end_ts=old + timedelta(seconds=90), o=1, h=1, l=1, c=1, v=1
    )
    agg._derived_store[old_key] = old_derived

    assert (ref, old) in agg._base_store
    assert old_key in agg._derived_store
    list(agg.run_gc(now=now))
    assert (ref, old) not in agg._base_store
    assert old_key not in agg._derived_store


def test_run_gc_evicts_incomplete_buckets():
    cfg = DirectFromBaseAggregatorConfig(
        source_name="x",
        base_timeframe=Timeframe(30, TimeframeUnit.SECOND),
        settle_seconds=0,
        keep_closed_buckets_seconds=0,
        keep_base_bars_seconds=0,
        keep_derived_bars_seconds=0,
        evict_incomplete_buckets_seconds=10,
    )
    agg = DirectFromBaseAggregator(cfg)

    ref = InstrumentRef("AAPL")
    base_tf = Timeframe(30, TimeframeUnit.SECOND)
    derived_tf = Timeframe(90, TimeframeUnit.SECOND)
    now = datetime(2021, 1, 1, 10, 0, tzinfo=timezone.utc)
    old = now - timedelta(seconds=20)

    key = BarKey(ref=ref, tf=derived_tf, start_ts=old)
    bucket = _BucketAgg(key, base_tf)
    bucket._last_updated_ts = old
    agg._bucket_aggs[key] = bucket

    assert key in agg._bucket_aggs
    list(agg.run_gc(now=now))
    assert key not in agg._bucket_aggs


def test_batch_subscribe_auto_subscribe_constituents():
    cfg = DirectFromBaseAggregatorConfig(
        source_name="x", base_timeframe=Timeframe(30, TimeframeUnit.SECOND)
    )
    agg = DirectFromBaseAggregator(cfg)

    tf = Timeframe(30, TimeframeUnit.SECOND)
    aapl = InstrumentRef("AAPL")
    msft = InstrumentRef("MSFT")

    msg = MDBarBatchSubscribeRequest(
        ts=datetime.now(timezone.utc).timestamp(),
        source="src",
        instrument_refs=(msft, aapl),
        timeframe=tf,
        auto_subscribe_constituents=True,
    )
    agg.on_subscribe_batch(msg)

    assert agg.subscribed_timeframes(aapl) == (tf,)
    assert agg.subscribed_timeframes(msft) == (tf,)
    assert tf in agg._batch_subs
    assert (aapl, msft) in agg._batch_subs[tf]


def test_batch_completed_emits_for_base_pass_through():
    cfg = DirectFromBaseAggregatorConfig(
        source_name="x", base_timeframe=Timeframe(30, TimeframeUnit.SECOND)
    )
    agg = DirectFromBaseAggregator(cfg)

    tf = Timeframe(30, TimeframeUnit.SECOND)
    aapl = InstrumentRef("AAPL")
    msft = InstrumentRef("MSFT")
    t0 = datetime(2021, 1, 1, 9, 30, tzinfo=timezone.utc)

    msg = MDBarBatchSubscribeRequest(
        ts=t0.timestamp(),
        source="src",
        instrument_refs=(aapl, msft),
        timeframe=tf,
        auto_subscribe_constituents=True,
    )
    agg.on_subscribe_batch(msg)

    bar_a = OHLCVBar(
        start_ts=t0, end_ts=t0 + timedelta(seconds=30), o=1, h=2, l=1, c=1.5, v=10
    )
    bar_m = OHLCVBar(
        start_ts=t0, end_ts=t0 + timedelta(seconds=30), o=2, h=3, l=1.5, c=2.5, v=20
    )

    ev_a = BaseBarUpserted(
        ts=t0.timestamp(),
        source="src",
        key=BarKey(ref=aapl, tf=tf, start_ts=t0),
        curr=bar_a,
        prev=None,
        is_correction=False,
    )
    ev_m = BaseBarUpserted(
        ts=t0.timestamp(),
        source="src",
        key=BarKey(ref=msft, tf=tf, start_ts=t0),
        curr=bar_m,
        prev=None,
        is_correction=False,
    )

    outs_a = list(agg.on_base_upsert(ev_a, now=t0))
    assert any(isinstance(ev, BarCompleted) for ev in outs_a)
    assert not any(isinstance(ev, BarBatchCompleted) for ev in outs_a)

    outs_m = list(agg.on_base_upsert(ev_m, now=t0))
    assert any(isinstance(ev, BarCompleted) for ev in outs_m)
    batch_events = [ev for ev in outs_m if isinstance(ev, BarBatchCompleted)]
    assert len(batch_events) == 1
    assert batch_events[0].key.refs == (aapl, msft)
    assert batch_events[0].key.tf == tf
    assert batch_events[0].key.start_ts == t0


def test_batch_completed_emits_for_derived_timeframe():
    cfg = DirectFromBaseAggregatorConfig(
        source_name="x", base_timeframe=Timeframe(30, TimeframeUnit.SECOND)
    )
    agg = DirectFromBaseAggregator(cfg)

    base_tf = Timeframe(30, TimeframeUnit.SECOND)
    derived_tf = Timeframe(90, TimeframeUnit.SECOND)
    aapl = InstrumentRef("AAPL")
    msft = InstrumentRef("MSFT")

    t0 = datetime(2021, 1, 1, 9, 30, tzinfo=timezone.utc)
    t1 = t0 + timedelta(seconds=30)
    t2 = t0 + timedelta(seconds=60)

    msg = MDBarBatchSubscribeRequest(
        ts=t0.timestamp(),
        source="src",
        instrument_refs=(aapl, msft),
        timeframe=derived_tf,
        auto_subscribe_constituents=True,
    )
    agg.on_subscribe_batch(msg)

    def make_ev(ref: InstrumentRef, ts: datetime, o: float) -> BaseBarUpserted:
        bar = OHLCVBar(
            start_ts=ts,
            end_ts=ts + timedelta(seconds=30),
            o=o,
            h=o + 1,
            l=o - 0.5,
            c=o + 0.25,
            v=10,
        )
        return BaseBarUpserted(
            ts=ts.timestamp(),
            source="src",
            key=BarKey(ref=ref, tf=base_tf, start_ts=ts),
            curr=bar,
            prev=None,
            is_correction=False,
        )

    assert list(agg.on_base_upsert(make_ev(aapl, t0, 1.0), now=t0)) == []
    assert list(agg.on_base_upsert(make_ev(aapl, t1, 1.5), now=t1)) == []
    outs_a = list(agg.on_base_upsert(make_ev(aapl, t2, 2.0), now=t2))
    assert any(isinstance(ev, BarCompleted) for ev in outs_a)
    assert not any(isinstance(ev, BarBatchCompleted) for ev in outs_a)

    assert list(agg.on_base_upsert(make_ev(msft, t0, 2.0), now=t0)) == []
    assert list(agg.on_base_upsert(make_ev(msft, t1, 2.5), now=t1)) == []
    outs_m = list(agg.on_base_upsert(make_ev(msft, t2, 3.0), now=t2))
    assert any(isinstance(ev, BarCompleted) for ev in outs_m)
    batch_events = [ev for ev in outs_m if isinstance(ev, BarBatchCompleted)]
    assert len(batch_events) == 1
    assert batch_events[0].key.refs == (aapl, msft)
    assert batch_events[0].key.tf == derived_tf
    assert batch_events[0].key.start_ts == t0
