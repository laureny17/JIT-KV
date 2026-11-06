"""Loads Mooncake request traces."""

import json
from dataclasses import dataclass
from os import PathLike


@dataclass(frozen=True)
class Request:
    id: int
    arrival_s: float
    input_tokens: int
    output_tokens: int


def load_trace(path: str | PathLike[str], load_factor: float = 1.0) -> list[Request]:
    """Returns the trace's requests in arrival order.

    Trace timestamps are in milliseconds. A load_factor above 1 compresses
    arrival times, raising the request rate without changing request sizes.
    """
    if load_factor <= 0:
        raise ValueError(f"load_factor must be positive, got {load_factor}")

    with open(path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    records.sort(key=lambda record: record["timestamp"])

    return [
        Request(
            id=i,
            arrival_s=record["timestamp"] / 1000 / load_factor,
            input_tokens=record["input_length"],
            output_tokens=record["output_length"],
        )
        for i, record in enumerate(records)
    ]
