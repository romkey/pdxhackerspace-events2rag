import logging
from datetime import UTC, datetime, timedelta

import pytest

from events2rag.text_utils import (
    collapse_whitespace,
    estimate_frequency,
    human_date,
    human_duration,
    prepare_embedding_text,
    strip_html,
    temporal_status,
    truncate_for_embedding,
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


def test_estimate_frequency_biweekly() -> None:
    times = [
        datetime(2026, 3, 1, tzinfo=UTC),
        datetime(2026, 3, 12, tzinfo=UTC),
        datetime(2026, 3, 23, tzinfo=UTC),
    ]
    assert estimate_frequency(times) == "biweekly"


def test_estimate_frequency_every_three_weeks() -> None:
    times = [
        datetime(2026, 3, 1, tzinfo=UTC),
        datetime(2026, 3, 22, tzinfo=UTC),
        datetime(2026, 4, 12, tzinfo=UTC),
    ]
    assert estimate_frequency(times) == "every three weeks"


def test_estimate_frequency_monthly() -> None:
    times = [
        datetime(2026, 1, 1, tzinfo=UTC),
        datetime(2026, 2, 1, tzinfo=UTC),
        datetime(2026, 3, 1, tzinfo=UTC),
    ]
    assert estimate_frequency(times) == "monthly"


def test_estimate_frequency_bimonthly() -> None:
    times = [
        datetime(2026, 1, 1, tzinfo=UTC),
        datetime(2026, 3, 1, tzinfo=UTC),
        datetime(2026, 5, 1, tzinfo=UTC),
    ]
    assert estimate_frequency(times) == "bimonthly"


def test_estimate_frequency_one_time() -> None:
    assert estimate_frequency([datetime(2026, 3, 1, tzinfo=UTC)]) == "one-time"


def test_collapse_whitespace() -> None:
    assert collapse_whitespace("  hello   world  ") == "hello world"


def test_truncate_for_embedding_short_text_unchanged() -> None:
    text = "This is short"
    assert truncate_for_embedding(text, max_tokens=512) == text


def test_truncate_for_embedding_long_text_truncated() -> None:
    text = "word " * 1000
    result = truncate_for_embedding(text, max_tokens=10)
    assert len(result) <= 10 * 3
    assert not result.endswith(" ")


def test_truncate_for_embedding_logs_warning(caplog: pytest.LogCaptureFixture) -> None:
    text = "word " * 1000
    log = logging.getLogger("test_truncate")
    with caplog.at_level(logging.WARNING, logger="test_truncate"):
        truncate_for_embedding(text, max_tokens=10, logger=log)
    assert "Truncated" in caplog.text


def test_human_duration_minutes_only() -> None:
    start = datetime(2026, 3, 1, 18, 0, tzinfo=UTC)
    assert human_duration(start, start + timedelta(minutes=45)) == "45 minutes"


def test_human_duration_none_when_end_not_after_start() -> None:
    start = datetime(2026, 3, 1, 18, 0, tzinfo=UTC)
    assert human_duration(start, start) is None
    assert human_duration(start, start - timedelta(hours=1)) is None


def test_human_duration_days() -> None:
    start = datetime(2026, 3, 1, 18, 0, tzinfo=UTC)
    assert human_duration(start, start + timedelta(days=1)) == "1 day"
    assert human_duration(start, start + timedelta(days=2)) == "2 days"
    assert human_duration(start, start + timedelta(days=1, hours=1)) == "1 day, 1 hour"
    assert (
        human_duration(start, start + timedelta(days=2, hours=5)) == "2 days, 5 hours"
    )


def test_human_duration_single_hour() -> None:
    start = datetime(2026, 3, 1, 18, 0, tzinfo=UTC)
    assert human_duration(start, start + timedelta(hours=1)) == "1 hour"
    assert (
        human_duration(start, start + timedelta(hours=1, minutes=30))
        == "1 hour, 30 minutes"
    )


def test_estimate_frequency_daily() -> None:
    times = [datetime(2026, 3, day, tzinfo=UTC) for day in (1, 2, 3)]
    assert estimate_frequency(times) == "daily"


def test_estimate_frequency_occasional() -> None:
    times = [
        datetime(2026, 1, 1, tzinfo=UTC),
        datetime(2026, 6, 1, tzinfo=UTC),
    ]
    assert estimate_frequency(times) == "occasional"


def test_estimate_frequency_is_order_independent() -> None:
    times = [
        datetime(2026, 3, 15, tzinfo=UTC),
        datetime(2026, 3, 1, tzinfo=UTC),
        datetime(2026, 3, 8, tzinfo=UTC),
    ]
    assert estimate_frequency(times) == "weekly"


def test_prepare_embedding_text_reserves_room_for_prefix() -> None:
    result = prepare_embedding_text("word " * 100, "passage: ", max_tokens=20)
    assert result.startswith("passage: ")
    assert len(result) <= 20 * 3


def test_prepare_embedding_text_without_prefix_just_truncates() -> None:
    text = "word " * 100
    assert prepare_embedding_text(text, "", 10) == truncate_for_embedding(text, 10)
