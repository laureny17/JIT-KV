"""Simulates requests moving from prefill to the end of decode.

The simulator provides the steps a request can take, such as prefill or waiting
for a decode GPU slot. A strategy (see strategies.py) lists which steps a request
takes and in what order. Each step is a generator that pauses at every `yield`
until the simulated time or GPU resources it needs are available.

Requests compete only for decode GPU slots and memory.
"""

from collections import deque
from collections.abc import Callable, Generator, Iterable
from dataclasses import dataclass

from jitkv.events import EventLoop
from jitkv.trace import Request

Resume = Callable[[], None]
Pause = Callable[[Resume], None]
Steps = Generator[Pause, None, None]


@dataclass(frozen=True)
class SimConfig:
    # Llama-3-70B-like: 2 (K, V) * 80 layers * 8 KV heads * 128 head dim * 2 bytes.
    kv_bytes_per_token: int = 327_680
    # 100 Gbps network and an effective PCIe Gen4 x16 link.
    network_bytes_per_s: float = 12.5e9
    pcie_bytes_per_s: float = 30e9
    # Placeholders until we calibrate them against published measurements.
    prefill_tokens_per_s: float = 10_000
    decode_s_per_token: float = 0.04
    decode_slots: int = 16
    # An 80 GB GPU, ignoring the memory the model itself occupies.
    decode_memory_bytes: int = 80_000_000_000


class DecodeGpu:
    """Tracks the slots and memory that admitted requests hold."""

    def __init__(self, slots: int, memory_bytes: int) -> None:
        self._free_slots = slots
        self._free_memory_bytes = memory_bytes

    def can_reserve(self, kv_bytes: int) -> bool:
        return self._free_slots > 0 and kv_bytes <= self._free_memory_bytes

    def reserve(self, kv_bytes: int) -> None:
        if not self.can_reserve(kv_bytes):
            raise RuntimeError(f"no room to reserve {kv_bytes} bytes")
        self._free_slots -= 1
        self._free_memory_bytes -= kv_bytes

    def release(self, kv_bytes: int) -> None:
        self._free_slots += 1
        self._free_memory_bytes += kv_bytes


@dataclass
class RequestRecord:
    """The time a request reached each point on its path.

    A field stays None if the request's strategy skips that point.
    """

    request: Request
    # When the request got its GPU slot and memory.
    admitted_s: float | None = None
    prefill_end_s: float | None = None
    transfer_end_s: float | None = None
    copy_end_s: float | None = None
    # Start of decode, the point TTFT is measured to.
    decode_start_s: float | None = None
    finish_s: float | None = None

    @property
    def ttft_s(self) -> float:
        """Time from arrival until the request can start decoding."""
        return self.decode_start_s - self.request.arrival_s


@dataclass
class _Waiter:
    kv_bytes: int
    resume: Resume


class Simulator:
    def __init__(
        self, config: SimConfig, strategy: Callable[["Simulator", RequestRecord], Steps]
    ) -> None:
        self._config = config
        self._strategy = strategy
        self._loop = EventLoop()
        self._gpu = DecodeGpu(config.decode_slots, config.decode_memory_bytes)
        self._waiting: deque[_Waiter] = deque()

    def run(self, requests: Iterable[Request]) -> list[RequestRecord]:
        records = [RequestRecord(request) for request in requests]
        for record in records:
            if self._kv_bytes(record) > self._config.decode_memory_bytes:
                raise ValueError(
                    f"request {record.request.id} can never fit in decode GPU memory"
                )
        for record in records:
            self._loop.schedule_at(
                record.request.arrival_s, lambda r=record: self._start(r)
            )
        self._loop.run()
        return records

    # The stages a request can go through. Strategies choose their order.

    def wait_for_slot_and_memory(self, record: RequestRecord) -> Steps:
        yield self._wait_in_line(self._kv_bytes(record))
        record.admitted_s = self._loop.now

    def prefill(self, record: RequestRecord) -> Steps:
        yield self._sleep(record.request.input_tokens / self._config.prefill_tokens_per_s)
        record.prefill_end_s = self._loop.now

    def network_transfer(self, record: RequestRecord) -> Steps:
        yield self._sleep(self._kv_bytes(record) / self._config.network_bytes_per_s)
        record.transfer_end_s = self._loop.now

    def copy_to_gpu(self, record: RequestRecord) -> Steps:
        yield self._sleep(self._kv_bytes(record) / self._config.pcie_bytes_per_s)
        record.copy_end_s = self._loop.now

    def decode(self, record: RequestRecord) -> Steps:
        record.decode_start_s = self._loop.now
        yield self._sleep(record.request.output_tokens * self._config.decode_s_per_token)
        record.finish_s = self._loop.now
        self._gpu.release(self._kv_bytes(record))
        self._admit_waiting()

    # Internals.

    def _start(self, record: RequestRecord) -> None:
        self._advance(self._strategy(self, record))

    def _advance(self, steps: Steps) -> None:
        """Runs a request's steps until the next pause, then waits to resume."""
        try:
            pause = next(steps)
        except StopIteration:
            return
        pause(lambda: self._advance(steps))

    def _sleep(self, delay_s: float) -> Pause:
        def pause(resume: Resume) -> None:
            self._loop.schedule_at(self._loop.now + delay_s, resume)

        return pause

    def _wait_in_line(self, kv_bytes: int) -> Pause:
        def pause(resume: Resume) -> None:
            self._waiting.append(_Waiter(kv_bytes, resume))
            self._admit_waiting()

        return pause

    def _admit_waiting(self) -> None:
        # Admit in the order requests joined the line. A smaller request never
        # overtakes the request at the front, even if it would fit.
        while self._waiting and self._gpu.can_reserve(self._waiting[0].kv_bytes):
            waiter = self._waiting.popleft()
            self._gpu.reserve(waiter.kv_bytes)
            waiter.resume()

    def _kv_bytes(self, record: RequestRecord) -> int:
        return record.request.input_tokens * self._config.kv_bytes_per_token
