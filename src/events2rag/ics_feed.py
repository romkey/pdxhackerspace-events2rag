from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import recurring_ical_events
import requests
from icalendar import Calendar

from events2rag.datetime_utils import ensure_utc
from events2rag.models import EventOccurrence
from events2rag.text_utils import collapse_whitespace, strip_html


def fetch_ics(url: str, timeout_seconds: int) -> str:
    response = requests.get(url, timeout=timeout_seconds)
    response.raise_for_status()
    return response.text


def parse_ics_occurrences(
    content: str,
    lookback_days: int = 30,
    lookahead_days: int = 365,
    feed_url: str | None = None,
) -> list[EventOccurrence]:
    calendar = Calendar.from_ical(content)
    start = datetime.now(tz=UTC) - timedelta(days=lookback_days)
    end = datetime.now(tz=UTC) + timedelta(days=lookahead_days)
    events = recurring_ical_events.of(calendar).between(start, end)
    occurrences: list[EventOccurrence] = []
    for event in events:
        occurrence = _event_to_occurrence(event, feed_url=feed_url)
        if occurrence:
            occurrences.append(occurrence)
    return occurrences


def _event_to_occurrence(
    event: Any, feed_url: str | None = None
) -> EventOccurrence | None:
    raw_start = event.get("DTSTART")
    if raw_start is None:
        return None
    start = _to_datetime(raw_start.dt)
    raw_end = event.get("DTEND")
    end = _to_datetime(raw_end.dt) if raw_end else None
    uid = str(event.get("UID") or "ics-event")
    occurrence_id = f"{uid}:{start.isoformat()}"
    url = _safe_str(event.get("URL")) or feed_url
    title = _safe_str(event.get("SUMMARY")) or uid
    raw_description = _safe_str(event.get("DESCRIPTION")) or ""
    description = collapse_whitespace(strip_html(raw_description))
    location = _safe_str(event.get("LOCATION"))
    categories = event.get("CATEGORIES")
    tags: list[str] = []
    if categories:
        tags = [str(x) for x in categories.cats]

    return EventOccurrence(
        occurrence_id=occurrence_id,
        event_id=uid,
        title=title,
        description=description,
        start_time=start,
        end_time=end,
        location=location,
        source_url=url,
        tags=tags,
        source_type="ics",
        last_modified=_to_datetime(event.get("LAST-MODIFIED").dt)
        if event.get("LAST-MODIFIED")
        else None,
    )


def _safe_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _to_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return ensure_utc(value)
    return datetime(value.year, value.month, value.day, tzinfo=UTC)
