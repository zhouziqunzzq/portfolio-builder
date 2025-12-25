import asyncio
import logging
from typing import Any, Dict, Optional, Set
from .events import BaseEvent
from .topic import Topic


class Subscription:
    """
    A subscription has its own inbox queue. The service consumes from this queue.
    """

    def __init__(self, bus: "EventBus", topics: Set[Topic], maxsize: int):
        self._bus = bus
        self.topics = topics
        self.q: asyncio.Queue[BaseEvent] = asyncio.Queue(maxsize=maxsize)
        self._closed = False

    async def next(self) -> BaseEvent:
        return await self.q.get()

    def task_done(self) -> None:
        self.q.task_done()

    async def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._bus.unsubscribe(self)


class EventBus:
    """
    Pub-sub event bus.

    - Each subscriber gets its own queue.
    - Publishing an event delivers it to all subscribers who subscribed to that topic
      or to the wildcard Topic.LOG/Topic.STOP etc as they choose.
    - Certain topics can be configured as true broadcasts: publish once -> everyone sees it.
    """

    def __init__(
        self,
        per_subscriber_queue_size: int = 10_000,
        drop_if_full: bool = False,
        broadcast_topics: Optional[Set[Topic]] = None,
    ):
        self._subs_by_topic: Dict[Topic, Set[Subscription]] = {}
        self._all_subs: Set[Subscription] = set()
        self._default_qsize = per_subscriber_queue_size
        self._drop_if_full = drop_if_full
        self._broadcast_topics = frozenset(broadcast_topics or {Topic.STOP})
        self.log = logging.getLogger(self.__class__.__name__)

    def subscribe(
        self, topics: Set[Topic], maxsize: Optional[int] = None
    ) -> Subscription:
        sub = Subscription(self, topics=topics, maxsize=maxsize or self._default_qsize)
        self._all_subs.add(sub)
        for t in topics:
            self._subs_by_topic.setdefault(t, set()).add(sub)
        return sub

    def unsubscribe(self, sub: Subscription) -> None:
        if sub in self._all_subs:
            self._all_subs.remove(sub)
        for t in list(sub.topics):
            s = self._subs_by_topic.get(t)
            if s:
                s.discard(sub)
                if not s:
                    self._subs_by_topic.pop(t, None)

    async def publish(self, e: BaseEvent) -> None:
        # Broadcast topics: once published, every subscriber sees it regardless of
        # which topics they subscribed to.
        if e.topic in self._broadcast_topics:
            subs = self._all_subs
        else:
            subs = self._subs_by_topic.get(e.topic, set())
        if not subs:
            return

        # Fan-out. Two modes:
        # - block until each subscriber queue has space (drop_if_full=False)
        # - drop for slow subscribers (drop_if_full=True)
        for sub in subs:
            if self._drop_if_full:
                try:
                    sub.q.put_nowait(e)
                except asyncio.QueueFull:
                    self.log.warning(
                        f"Dropping event {e} for subscriber {sub} due to full queue"
                    )
            else:
                await sub.q.put(e)
