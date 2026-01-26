from dataclasses import dataclass, field
from typing import List

from algotrading.lib.types import AccountSnapshot, PositionSnapshot

from .topic import Topic
from .base import *


@dataclass(frozen=True, kw_only=True)
class AccountSnapshotEvent(BaseEvent):
    """Event containing broker account + positions snapshot.

    Intended to be published periodically by EML services.
    """

    topic: Topic = field(default=Topic.EXEC_ACCOUNT_SNAPSHOT, init=False)

    account: AccountSnapshot
    positions: List[PositionSnapshot] = field(default_factory=list)
