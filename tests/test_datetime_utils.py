from datetime import UTC, datetime, timedelta, timezone

from events2rag.datetime_utils import ensure_utc, parse_datetime


def test_parse_datetime_none() -> None:
    assert parse_datetime(None) is None


def test_parse_datetime_unsupported_type() -> None:
    assert parse_datetime(12345) is None  # type: ignore[arg-type]


def test_parse_datetime_string_with_offset_converts_to_utc() -> None:
    result = parse_datetime("2026-03-01T10:00:00-08:00")
    assert result == datetime(2026, 3, 1, 18, 0, tzinfo=UTC)
    assert result.tzinfo is UTC


def test_parse_datetime_naive_string_assumed_utc() -> None:
    result = parse_datetime("2026-03-01 10:00")
    assert result == datetime(2026, 3, 1, 10, 0, tzinfo=UTC)


def test_parse_datetime_datetime_passthrough() -> None:
    aware = datetime(2026, 3, 1, 10, 0, tzinfo=timezone(timedelta(hours=2)))
    assert parse_datetime(aware) == datetime(2026, 3, 1, 8, 0, tzinfo=UTC)


def test_ensure_utc_naive() -> None:
    naive = datetime(2026, 3, 1, 10, 0)  # noqa: DTZ001 - testing naive input
    assert ensure_utc(naive) == datetime(2026, 3, 1, 10, 0, tzinfo=UTC)
