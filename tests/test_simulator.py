import pytest

from jitkv.simulator import SimConfig, Simulator
from jitkv.trace import Request

# Chosen so that a 1,000-token prompt with 10 output tokens spends a distinct
# time in each stage: 1 s prefill, 2 s transfer, 4 s copy, 8 s decode.
CONFIG = SimConfig(
    kv_bytes_per_token=1_000,
    prefill_tokens_per_s=1_000,
    network_bytes_per_s=500_000,
    pcie_bytes_per_s=250_000,
    decode_s_per_token=0.8,
)


def request(id=0, arrival_s=0.0):
    return Request(id=id, arrival_s=arrival_s, input_tokens=1_000, output_tokens=10)


def test_request_passes_through_each_stage_in_order():
    [record] = Simulator(CONFIG).run([request(arrival_s=100.0)])

    assert record.prefill_end_s == pytest.approx(101.0)
    assert record.transfer_end_s == pytest.approx(103.0)
    assert record.copy_end_s == pytest.approx(107.0)
    assert record.decode_start_s == pytest.approx(107.0)
    assert record.finish_s == pytest.approx(115.0)


def test_ttft_is_prefill_plus_transfer_plus_copy():
    [record] = Simulator(CONFIG).run([request(arrival_s=100.0)])

    assert record.ttft_s == pytest.approx(1.0 + 2.0 + 4.0)


def test_requests_do_not_affect_each_other_without_contention():
    records = Simulator(CONFIG).run([request(id=0), request(id=1, arrival_s=0.5)])

    assert [r.request.id for r in records] == [0, 1]
    assert [r.ttft_s for r in records] == pytest.approx([7.0, 7.0])
