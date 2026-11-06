"""Simulates each request's path from prefill to the end of decode.

Stages do not contend for resources yet, so requests run independently.
"""

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


@dataclass
class RequestRecord:
    """The time each stage of a request ended."""

    request: Request
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


class Simulator:
    def __init__(self, config: SimConfig) -> None:
        self._config = config
        self._loop = EventLoop()

    def run(self, requests: Iterable[Request]) -> list[RequestRecord]:
        records = [RequestRecord(request) for request in requests]
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
        self._after(duration, lambda: self._start_copy(record))

    def _start_copy(self, record: RequestRecord) -> None:
        record.transfer_end_s = self._loop.now
        duration = self._kv_bytes(record) / self._config.pcie_bytes_per_s
        self._after(duration, lambda: self._start_decode(record))

    def _start_decode(self, record: RequestRecord) -> None:
        record.copy_end_s = self._loop.now
        record.decode_start_s = self._loop.now
        duration = record.request.output_tokens * self._config.decode_s_per_token
        self._after(duration, lambda: self._finish(record))

    def _finish(self, record: RequestRecord) -> None:
        record.finish_s = self._loop.now
