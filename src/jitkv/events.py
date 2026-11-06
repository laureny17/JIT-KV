"""A minimal discrete-event loop."""

import heapq
from collections.abc import Callable
from itertools import count


class EventLoop:
    def __init__(self) -> None:
        self.now = 0.0
        self._queue: list[tuple[float, int, Callable[[], None]]] = []
        self._sequence = count()

    def schedule_at(self, time: float, callback: Callable[[], None]) -> None:
        """Runs callback at the given time. Events at equal times run in FIFO order."""
        if time < self.now:
            raise ValueError(f"cannot schedule at {time}, current time is {self.now}")
        heapq.heappush(self._queue, (time, next(self._sequence), callback))

    def run(self) -> None:
        while self._queue:
            self.now, _, callback = heapq.heappop(self._queue)
            callback()
