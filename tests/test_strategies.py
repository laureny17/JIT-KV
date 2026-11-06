from dataclasses import replace

import pytest

from jitkv.simulator import SimConfig, Simulator
from jitkv.strategies import cpu_buffered_pull, direct_pull
from jitkv.trace import Request

# A 1,000-token prompt with 10 output tokens takes:
# 1 s prefill, 2 s network transfer, 4 s CPU-to-GPU copy, 8 s decode.
CONFIG = SimConfig(
    kv_bytes_per_token=1_000,
    prefill_tokens_per_s=1_000,
    network_bytes_per_s=500_000,
    pcie_bytes_per_s=250_000,
    decode_s_per_token=0.8,
)


def two_requests():
    return [
        Request(id=0, arrival_s=0.0, input_tokens=1_000, output_tokens=10),
        Request(id=1, arrival_s=0.5, input_tokens=1_000, output_tokens=10),
    ]


def run_one(strategy):
    request = Request(id=0, arrival_s=0.0, input_tokens=1_000, output_tokens=10)
    [record] = Simulator(CONFIG, strategy).run([request])
    return record


def test_direct_pull_reserves_after_prefill_and_skips_the_cpu_copy():
    record = run_one(direct_pull)

    assert record.admitted_s == pytest.approx(1.0)
    assert record.decode_start_s == pytest.approx(3.0)
    assert record.copy_end_s is None


def test_cpu_buffered_pull_reserves_after_transfer_and_copies():
    record = run_one(cpu_buffered_pull)

    assert record.admitted_s == pytest.approx(3.0)
    assert record.decode_start_s == pytest.approx(7.0)


def test_with_one_slot_direct_pull_transfers_after_the_slot_frees():
    _, second = Simulator(replace(CONFIG, decode_slots=1), direct_pull).run(two_requests())

    # The first request finishes at t=11, then the second spends 2 s transferring.
    assert second.admitted_s == pytest.approx(11.0)
    assert second.decode_start_s == pytest.approx(13.0)
    assert second.ttft_s == pytest.approx(12.5)


def test_with_one_slot_cpu_buffered_pull_copies_after_the_slot_frees():
    _, second = Simulator(replace(CONFIG, decode_slots=1), cpu_buffered_pull).run(
        two_requests()
    )

    # The first request finishes at t=15, then the second spends 4 s copying.
    assert second.admitted_s == pytest.approx(15.0)
    assert second.decode_start_s == pytest.approx(19.0)
    assert second.ttft_s == pytest.approx(18.5)
