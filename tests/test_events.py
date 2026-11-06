import pytest

from jitkv.events import EventLoop


def test_runs_events_in_time_order_and_advances_clock():
    loop = EventLoop()
    seen = []
    loop.schedule_at(2.0, lambda: seen.append(("late", loop.now)))
    loop.schedule_at(1.0, lambda: seen.append(("early", loop.now)))

    loop.run()

    assert seen == [("early", 1.0), ("late", 2.0)]


def test_equal_times_run_in_fifo_order():
    loop = EventLoop()
    seen = []
    loop.schedule_at(1.0, lambda: seen.append("first"))
    loop.schedule_at(1.0, lambda: seen.append("second"))

    loop.run()

    assert seen == ["first", "second"]


def test_events_can_schedule_more_events():
    loop = EventLoop()
    seen = []
    loop.schedule_at(1.0, lambda: loop.schedule_at(3.0, lambda: seen.append(loop.now)))

    loop.run()

    assert seen == [3.0]


def test_rejects_scheduling_in_the_past():
    loop = EventLoop()
    loop.schedule_at(1.0, lambda: None)
    loop.run()

    with pytest.raises(ValueError):
        loop.schedule_at(0.5, lambda: None)
