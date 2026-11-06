"""The KV-cache transfer strategies being compared.

Each strategy lists the steps a request takes, in order. Together, the order
decides how long GPU memory and slots sit reserved before decode can use them.
"""

from jitkv.simulator import RequestRecord, Simulator, Steps


def cpu_buffered_pull(sim: Simulator, record: RequestRecord) -> Steps:
    """Moves the KV cache to CPU memory, then copies it once the GPU has room.

    It reserves nothing early, and the reserved slot sits idle only during the
    CPU-to-GPU copy.
    """
    yield from sim.prefill(record)
    yield from sim.network_transfer(record)
    yield from sim.wait_for_slot_and_memory(record)
    yield from sim.copy_to_gpu(record)
    yield from sim.decode(record)
