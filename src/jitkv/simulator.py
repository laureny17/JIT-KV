"""Simulates each request's path from prefill to the end of decode.

Requests compete only for decode GPU slots and memory. Each request waits in
the CPU buffer until both are free, in the order its transfer finished.
"""

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass

from jitkv.events import EventLoop
from jitkv.trace import Request


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
    """The time a request reached each point on its path."""

    request: Request
    prefill_end_s: float | None = None
    transfer_end_s: float | None = None
    # When the request got its GPU slot and memory, which also starts the copy.
    admitted_s: float | None = None
    copy_end_s: float | None = None
    # Start of decode, the point TTFT is measured to.
    decode_start_s: float | None = None
    finish_s: float | None = None

    @property
    def ttft_s(self) -> float:
        """Time from arrival until the request can start decoding."""
        return self.decode_start_s - self.request.arrival_s


class Simulator:
    def __init__(self, config: SimConfig) -> None:
        self._config = config
        self._loop = EventLoop()
        self._gpu = DecodeGpu(config.decode_slots, config.decode_memory_bytes)
        self._waiting: deque[RequestRecord] = deque()

    def run(self, requests: Iterable[Request]) -> list[RequestRecord]:
        records = [RequestRecord(request) for request in requests]
        for record in records:
            if self._kv_bytes(record) > self._config.decode_memory_bytes:
                raise ValueError(
                    f"request {record.request.id} can never fit in decode GPU memory"
                )
        for record in records:
            self._loop.schedule_at(
                record.request.arrival_s, lambda r=record: self._start_prefill(r)
            )
        self._loop.run()
        return records

    def _after(self, delay_s: float, callback) -> None:
        self._loop.schedule_at(self._loop.now + delay_s, callback)

    def _kv_bytes(self, record: RequestRecord) -> int:
        return record.request.input_tokens * self._config.kv_bytes_per_token

    def _start_prefill(self, record: RequestRecord) -> None:
        duration = record.request.input_tokens / self._config.prefill_tokens_per_s
        self._after(duration, lambda: self._start_transfer(record))

    def _start_transfer(self, record: RequestRecord) -> None:
        record.prefill_end_s = self._loop.now
        duration = self._kv_bytes(record) / self._config.network_bytes_per_s
        self._after(duration, lambda: self._wait_for_admission(record))

    def _wait_for_admission(self, record: RequestRecord) -> None:
        record.transfer_end_s = self._loop.now
        self._waiting.append(record)
        self._admit_waiting()

    def _admit_waiting(self) -> None:
        # Admit in the order transfers finish. A smaller request never overtakes
        # the request at the front of the line, even if it would fit.
        while self._waiting and self._gpu.can_reserve(self._kv_bytes(self._waiting[0])):
            self._start_copy(self._waiting.popleft())

    def _start_copy(self, record: RequestRecord) -> None:
        kv_bytes = self._kv_bytes(record)
        self._gpu.reserve(kv_bytes)
        record.admitted_s = self._loop.now
        duration = kv_bytes / self._config.pcie_bytes_per_s
        self._after(duration, lambda: self._start_decode(record))

    def _start_decode(self, record: RequestRecord) -> None:
        record.copy_end_s = self._loop.now
        record.decode_start_s = self._loop.now
        duration = record.request.output_tokens * self._config.decode_s_per_token
        self._after(duration, lambda: self._finish(record))

    def _finish(self, record: RequestRecord) -> None:
        record.finish_s = self._loop.now
        self._gpu.release(self._kv_bytes(record))
        self._admit_waiting()
