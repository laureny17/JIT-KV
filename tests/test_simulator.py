import random
from dataclasses import replace

import pytest

from jitkv.simulator import SimConfig, Simulator
from jitkv.strategies import cpu_buffered_pull
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

ALL_STRATEGIES = [cpu_buffered_pull]


def request(id=0, arrival_s=0.0):
    return Request(id=id, arrival_s=arrival_s, input_tokens=1_000, output_tokens=10)


def two_requests():
    return [request(id=0), request(id=1, arrival_s=0.5)]


def test_request_passes_through_each_stage_in_order():
    [record] = Simulator(CONFIG, cpu_buffered_pull).run([request(arrival_s=100.0)])

    assert record.prefill_end_s == pytest.approx(101.0)
    assert record.transfer_end_s == pytest.approx(103.0)
    assert record.copy_end_s == pytest.approx(107.0)
    assert record.decode_start_s == pytest.approx(107.0)
    assert record.finish_s == pytest.approx(115.0)


def test_ttft_is_prefill_plus_transfer_plus_copy():
    [record] = Simulator(CONFIG, cpu_buffered_pull).run([request(arrival_s=100.0)])

    assert record.ttft_s == pytest.approx(1.0 + 2.0 + 4.0)


def test_requests_do_not_affect_each_other_without_contention():
    records = Simulator(CONFIG, cpu_buffered_pull).run(two_requests())

    assert [r.request.id for r in records] == [0, 1]
    assert [r.ttft_s for r in records] == pytest.approx([7.0, 7.0])


def test_request_waits_for_a_free_slot():
    config = replace(CONFIG, decode_slots=1)

    first, second = Simulator(config, cpu_buffered_pull).run(two_requests())

    # The first request holds the slot until it finishes at t=15.
    assert first.finish_s == pytest.approx(15.0)
    assert second.transfer_end_s == pytest.approx(3.5)
    assert second.admitted_s == pytest.approx(15.0)
    assert second.decode_start_s == pytest.approx(19.0)
    assert second.ttft_s == pytest.approx(18.5)


def test_request_waits_for_free_memory():
    # Slots are plentiful, but memory holds only one 1 MB KV cache at a time.
    config = replace(CONFIG, decode_memory_bytes=1_500_000)

    _, second = Simulator(config, cpu_buffered_pull).run(two_requests())

    assert second.admitted_s == pytest.approx(15.0)


def test_smaller_request_does_not_overtake_the_front_of_the_line():
    config = replace(CONFIG, decode_memory_bytes=2_000_000)
    running = Request(id=0, arrival_s=0.0, input_tokens=1_000, output_tokens=10)
    large = Request(id=1, arrival_s=0.0, input_tokens=2_000, output_tokens=10)
    small = Request(id=2, arrival_s=5.0, input_tokens=500, output_tokens=10)

    _, large_record, small_record = Simulator(config, cpu_buffered_pull).run(
        [running, large, small]
    )

    # Large joins the line at t=6 and small at t=6.5. Small would fit next to
    # the running request, but it stays behind large until large finishes.
    assert large_record.admitted_s == pytest.approx(15.0)
    assert large_record.finish_s == pytest.approx(31.0)
    assert small_record.admitted_s == pytest.approx(31.0)


def test_rejects_request_larger_than_gpu_memory():
    config = replace(CONFIG, decode_memory_bytes=999_999)

    with pytest.raises(ValueError):
        Simulator(config, cpu_buffered_pull).run([request()])


@pytest.mark.parametrize("strategy", ALL_STRATEGIES)
def test_random_workload_finishes_within_gpu_limits(strategy):
    rng = random.Random(0)
    config = replace(CONFIG, decode_slots=4, decode_memory_bytes=5_000_000)
    requests = [
        Request(
            id=i,
            arrival_s=rng.uniform(0, 100),
            input_tokens=rng.randint(100, 3_000),
            output_tokens=rng.randint(1, 50),
        )
        for i in range(500)
    ]

    records = Simulator(config, strategy).run(requests)

    assert all(record.finish_s is not None for record in records)

    # Replay when each request held GPU resources, releases before reservations
    # so that a request admitted the moment another finishes is not double-counted.
    changes = []
    for record in records:
        kv_bytes = record.request.input_tokens * config.kv_bytes_per_token
        changes.append((record.admitted_s, 1, 1, kv_bytes))
        changes.append((record.finish_s, 0, -1, -kv_bytes))
    slots_used = memory_used = 0
    for _, _, slot_change, memory_change in sorted(changes):
        slots_used += slot_change
        memory_used += memory_change
        assert slots_used <= config.decode_slots
        assert memory_used <= config.decode_memory_bytes
