import json

import pytest

from jitkv.trace import Request, load_trace


@pytest.fixture
def trace_path(tmp_path):
    path = tmp_path / "trace.jsonl"
    records = [
        {"timestamp": 3000, "input_length": 200, "output_length": 20, "hash_ids": [2]},
        {"timestamp": 1000, "input_length": 100, "output_length": 10, "hash_ids": [1]},
    ]
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    return path


def test_converts_records_and_orders_by_arrival(trace_path):
    assert load_trace(trace_path) == [
        Request(id=0, arrival_s=1.0, input_tokens=100, output_tokens=10),
        Request(id=1, arrival_s=3.0, input_tokens=200, output_tokens=20),
    ]


def test_load_factor_compresses_arrivals_only(trace_path):
    requests = load_trace(trace_path, load_factor=2.0)

    assert [r.arrival_s for r in requests] == [0.5, 1.5]
    assert [r.input_tokens for r in requests] == [100, 200]


@pytest.mark.parametrize("load_factor", [0, -1])
def test_rejects_non_positive_load_factor(trace_path, load_factor):
    with pytest.raises(ValueError):
        load_trace(trace_path, load_factor=load_factor)
