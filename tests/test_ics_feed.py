from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import requests

from events2rag import ics_feed
from events2rag.ics_feed import (
    _event_to_occurrence,
    _to_datetime,
    fetch_ics,
    parse_ics_occurrences,
)


def _ics(*events: str) -> str:
    body = "\r\n".join(events)
    return (
        "BEGIN:VCALENDAR\r\n"
        "VERSION:2.0\r\n"
        "PRODID:-//events2rag//tests//EN\r\n"
        f"{body}\r\n"
        "END:VCALENDAR\r\n"
    )


def _fmt(dt: datetime) -> str:
    return dt.strftime("%Y%m%dT%H%M%SZ")


def test_parse_ics_expands_recurring_event() -> None:
    start = (datetime.now(tz=UTC) + timedelta(days=1)).replace(
        hour=18, minute=0, second=0, microsecond=0
    )
    end = start + timedelta(hours=2)
    content = _ics(
        "BEGIN:VEVENT",
        "UID:weekly-1",
        f"DTSTART:{_fmt(start)}",
        f"DTEND:{_fmt(end)}",
        "RRULE:FREQ=WEEKLY;COUNT=3",
        "SUMMARY:Weekly Hack Night",
        "DESCRIPTION:<p>Bring   a <b>project</b></p>",
        "LOCATION:Main Room",
        "URL:https://example.com/hack-night",
        "CATEGORIES:social,hacking",
        "LAST-MODIFIED:20260101T000000Z",
        "END:VEVENT",
    )

    occurrences = parse_ics_occurrences(
        content, lookback_days=1, lookahead_days=60
    )

    assert len(occurrences) == 3
    first = sorted(occurrences, key=lambda o: o.start_time)[0]
    assert first.event_id == "weekly-1"
    assert first.occurrence_id == f"weekly-1:{start.isoformat()}"
    assert first.title == "Weekly Hack Night"
    assert first.description == "Bring a project"
    assert first.start_time == start
    assert first.end_time == end
    assert first.location == "Main Room"
    assert first.source_url == "https://example.com/hack-night"
    assert first.tags == ["social", "hacking"]
    assert first.source_type == "ics"
    assert first.last_modified == datetime(2026, 1, 1, tzinfo=UTC)


def test_parse_ics_respects_window_and_defaults() -> None:
    soon = (datetime.now(tz=UTC) + timedelta(days=2)).date()
    far = (datetime.now(tz=UTC) + timedelta(days=400)).date()
    content = _ics(
        "BEGIN:VEVENT",
        "UID:all-day",
        f"DTSTART;VALUE=DATE:{soon:%Y%m%d}",
        "END:VEVENT",
        "BEGIN:VEVENT",
        "UID:too-far",
        f"DTSTART;VALUE=DATE:{far:%Y%m%d}",
        "SUMMARY:Far Away",
        "END:VEVENT",
    )

    occurrences = parse_ics_occurrences(
        content, feed_url="https://example.com/feed.ics"
    )

    assert len(occurrences) == 1
    occ = occurrences[0]
    assert occ.title == "all-day"
    assert occ.description == ""
    assert occ.start_time == datetime(
        soon.year, soon.month, soon.day, tzinfo=UTC
    )
    assert occ.location is None
    assert occ.source_url == "https://example.com/feed.ics"
    assert occ.tags == []
    assert occ.last_modified is None


def test_event_without_dtstart_is_skipped() -> None:
    assert _event_to_occurrence({}) is None


def test_parse_ics_skips_events_without_occurrence(monkeypatch) -> None:
    monkeypatch.setattr(ics_feed, "_event_to_occurrence", lambda *a, **k: None)
    start = datetime.now(tz=UTC) + timedelta(days=1)
    content = _ics(
        "BEGIN:VEVENT",
        "UID:x",
        f"DTSTART:{_fmt(start)}",
        "END:VEVENT",
    )
    assert parse_ics_occurrences(content) == []


def test_to_datetime_handles_dates_and_datetimes() -> None:
    assert _to_datetime(date(2026, 5, 4)) == datetime(2026, 5, 4, tzinfo=UTC)
    assert _to_datetime(datetime(2026, 5, 4, 12, 30)) == datetime(
        2026, 5, 4, 12, 30, tzinfo=UTC
    )


def test_fetch_ics(monkeypatch) -> None:
    seen: dict = {}

    class FakeResponse:
        text = "BEGIN:VCALENDAR"

        def raise_for_status(self) -> None:
            seen["raised"] = True

    def fake_get(url, timeout=None):
        seen["url"] = url
        seen["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(requests, "get", fake_get)

    assert fetch_ics("https://example.com/feed.ics", 7) == "BEGIN:VCALENDAR"
    assert seen == {
        "url": "https://example.com/feed.ics",
        "timeout": 7,
        "raised": True,
    }
