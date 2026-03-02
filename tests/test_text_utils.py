from datetime import UTC, datetime

from events2rag.text_utils import (
    collapse_whitespace,
    estimate_frequency,
    human_date,
    human_duration,
    strip_html,
    temporal_status,
)


def test_strip_html_removes_tags() -> None:
    raw = "<p>Join us for <b>Soldering Night</b>!</p>"
    assert strip_html(raw) == "Join us for Soldering Night !"


def test_strip_html_passthrough_plain_text() -> None:
    assert strip_html("No HTML here") == "No HTML here"


def test_strip_html_empty() -> None:
    assert strip_html("") == ""


def test_human_date_includes_day_of_week() -> None:
    dt = datetime(2026, 3, 7, 18, 30, tzinfo=UTC)
    result = human_date(dt)
    assert "Saturday" in result
    assert "March" in result
    assert "2026" in result


def test_human_duration_hours() -> None:
    start = datetime(2026, 3, 7, 18, 0, tzinfo=UTC)
    end = datetime(2026, 3, 7, 20, 0, tzinfo=UTC)
    assert human_duration(start, end) == "2 hours"


def test_human_duration_hours_and_minutes() -> None:
    start = datetime(2026, 3, 7, 18, 0, tzinfo=UTC)
    end = datetime(2026, 3, 7, 19, 30, tzinfo=UTC)
    assert human_duration(start, end) == "1 hour, 30 minutes"


def test_human_duration_none_when_no_end() -> None:
    start = datetime(2026, 3, 7, 18, 0, tzinfo=UTC)
    assert human_duration(start, None) is None


def test_temporal_status_past() -> None:
    now = datetime(2026, 3, 10, tzinfo=UTC)
    start = datetime(2026, 3, 1, tzinfo=UTC)
    end = datetime(2026, 3, 1, 2, 0, tzinfo=UTC)
    assert temporal_status(start, end, now) == "past"


def test_temporal_status_current() -> None:
    now = datetime(2026, 3, 1, 1, 0, tzinfo=UTC)
    start = datetime(2026, 3, 1, tzinfo=UTC)
    end = datetime(2026, 3, 1, 2, 0, tzinfo=UTC)
    assert temporal_status(start, end, now) == "current"


def test_temporal_status_future() -> None:
    now = datetime(2026, 2, 28, tzinfo=UTC)
    start = datetime(2026, 3, 1, tzinfo=UTC)
    assert temporal_status(start, None, now) == "future"


def test_temporal_status_past_when_no_end() -> None:
    now = datetime(2026, 3, 2, tzinfo=UTC)
    start = datetime(2026, 3, 1, tzinfo=UTC)
    assert temporal_status(start, None, now) == "past"


def test_estimate_frequency_weekly() -> None:
    times = [
        datetime(2026, 3, 1, tzinfo=UTC),
        datetime(2026, 3, 8, tzinfo=UTC),
        datetime(2026, 3, 15, tzinfo=UTC),
    ]
    assert estimate_frequency(times) == "weekly"


def test_estimate_frequency_one_time() -> None:
    assert estimate_frequency([datetime(2026, 3, 1, tzinfo=UTC)]) == "one-time"


def test_collapse_whitespace() -> None:
    assert collapse_whitespace("  hello   world  ") == "hello world"
