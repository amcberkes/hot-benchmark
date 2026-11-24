"""Small rendezvous channel used by the EnergyPlus worker thread.

Kept independent of ``pyenergyplus`` so its lifecycle semantics can be unit
tested on machines without an EnergyPlus installation.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from threading import Condition
from typing import Generic, TypeVar, cast


T = TypeVar("T")


class ChannelClosed(RuntimeError):
    """Raised when communication is attempted after channel shutdown."""


@dataclass
class _Receiver(Generic[T]):
    ready: bool = False
    value: T | None = None


class Channel(Generic[T]):
    """Rendezvous delivery; closing wakes all waiting senders and receivers.

    A handoff committed before close remains deliverable. Operations that have
    not committed a handoff raise ``ChannelClosed`` when shutdown starts.
    """

    def __init__(self) -> None:
        self.closed = False
        self._condition = Condition()
        self._receivers: deque[_Receiver[T]] = deque()

    def _ensure_open(self) -> None:
        if self.closed:
            raise ChannelClosed("channel is closed")

    def put(self, value: T) -> None:
        with self._condition:
            self._condition.wait_for(lambda: self.closed or bool(self._receivers))
            self._ensure_open()
            receiver = self._receivers.popleft()
            receiver.value = value
            receiver.ready = True
            self._condition.notify_all()

    def get(self) -> T:
        with self._condition:
            self._ensure_open()
            receiver: _Receiver[T] = _Receiver()
            self._receivers.append(receiver)
            self._condition.notify_all()
            self._condition.wait_for(lambda: receiver.ready or self.closed)
            if receiver.ready:
                return cast(T, receiver.value)
            raise ChannelClosed("channel is closed")

    def close(self) -> None:
        """Cancel pending operations; repeated or racing calls are harmless."""
        with self._condition:
            self.closed = True
            self._receivers.clear()
            self._condition.notify_all()


__all__ = ["Channel", "ChannelClosed"]
