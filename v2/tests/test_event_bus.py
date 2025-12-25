from __future__ import annotations

import asyncio

from v2.src.events.event_bus import EventBus
from v2.src.events.events import BaseEvent
from v2.src.events.topic import Topic


def _run(coro):
    return asyncio.run(coro)


def test_publish_with_no_subscribers_is_noop():
    bus = EventBus()

    async def _case():
        await bus.publish(BaseEvent(topic=Topic.LOG, ts=1.0))

    _run(_case())


def test_subscriber_receives_published_event():
    bus = EventBus()
    sub = bus.subscribe({Topic.LOG})

    e = BaseEvent(topic=Topic.LOG, ts=123.0, source="unit")

    async def _case():
        await bus.publish(e)
        got = await asyncio.wait_for(sub.next(), timeout=0.25)
        assert got == e

    _run(_case())


def test_fanout_delivers_to_all_subscribers_of_topic():
    bus = EventBus()
    a = bus.subscribe({Topic.LOG})
    b = bus.subscribe({Topic.LOG})

    e = BaseEvent(topic=Topic.LOG, ts=1.0)

    async def _case():
        await bus.publish(e)
        got_a = await asyncio.wait_for(a.next(), timeout=0.25)
        got_b = await asyncio.wait_for(b.next(), timeout=0.25)
        assert got_a == e
        assert got_b == e

    _run(_case())


def test_unsubscribe_stops_delivery():
    bus = EventBus()
    sub = bus.subscribe({Topic.LOG})

    async def _case():
        await sub.close()
        await bus.publish(BaseEvent(topic=Topic.LOG, ts=1.0))
        assert sub.q.empty()  # should not receive after close

    _run(_case())


def test_drop_if_full_drops_slow_subscriber_events():
    bus = EventBus(per_subscriber_queue_size=1, drop_if_full=True)
    sub = bus.subscribe({Topic.LOG})

    e1 = BaseEvent(topic=Topic.LOG, ts=1.0)
    e2 = BaseEvent(topic=Topic.LOG, ts=2.0)

    async def _case():
        await bus.publish(e1)
        await bus.publish(e2)  # should be dropped because queue already full
        assert sub.q.qsize() == 1
        got = await asyncio.wait_for(sub.next(), timeout=0.25)
        assert got == e1
        assert sub.q.empty()

    _run(_case())


def test_blocking_mode_applies_backpressure_until_consumer_advances():
    bus = EventBus(per_subscriber_queue_size=1, drop_if_full=False)
    sub = bus.subscribe({Topic.LOG})

    e1 = BaseEvent(topic=Topic.LOG, ts=1.0)
    e2 = BaseEvent(topic=Topic.LOG, ts=2.0)

    async def _case():
        await bus.publish(e1)

        publish_task = asyncio.create_task(bus.publish(e2))
        await asyncio.sleep(0)  # let publish_task run
        assert not publish_task.done(), "Expected publish to block due to full queue"

        got1 = await asyncio.wait_for(sub.next(), timeout=0.25)
        assert got1 == e1

        await asyncio.wait_for(publish_task, timeout=0.25)
        got2 = await asyncio.wait_for(sub.next(), timeout=0.25)
        assert got2 == e2

    _run(_case())


def test_stop_is_broadcast_to_all_subscribers():
    bus = EventBus()

    # Subscribe to a non-STOP topic; should still receive STOP.
    sub_a = bus.subscribe({Topic.LOG})
    sub_b = bus.subscribe({Topic.MARKET_CLOCK})

    stop_evt = BaseEvent(topic=Topic.STOP, ts=99.0)

    async def _case():
        await bus.publish(stop_evt)
        got_a = await asyncio.wait_for(sub_a.next(), timeout=0.25)
        got_b = await asyncio.wait_for(sub_b.next(), timeout=0.25)
        assert got_a == stop_evt
        assert got_b == stop_evt

    _run(_case())


def test_broadcast_topics_can_be_customized():
    # Make LOG a broadcast topic, and disable STOP broadcast.
    bus = EventBus(broadcast_topics={Topic.LOG})

    # Neither subscribes to LOG, but both should still receive it due to broadcast.
    sub_a = bus.subscribe({Topic.MARKET_CLOCK})
    sub_b = bus.subscribe({Topic.BAR})

    log_evt = BaseEvent(topic=Topic.LOG, ts=1.0)
    stop_evt = BaseEvent(topic=Topic.STOP, ts=2.0)

    async def _case():
        await bus.publish(log_evt)
        got_a = await asyncio.wait_for(sub_a.next(), timeout=0.25)
        got_b = await asyncio.wait_for(sub_b.next(), timeout=0.25)
        assert got_a == log_evt
        assert got_b == log_evt

        # STOP is not broadcast in this configuration; subscribers should not receive it.
        await bus.publish(stop_evt)
        assert sub_a.q.empty()
        assert sub_b.q.empty()

    _run(_case())
