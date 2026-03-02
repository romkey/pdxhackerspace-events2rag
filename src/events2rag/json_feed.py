from __future__ import annotations

from typing import Any

import requests

from events2rag.datetime_utils import parse_datetime
from events2rag.models import EventOccurrence


def fetch_json(url: str, timeout_seconds: int) -> Any:
    response = requests.get(url, timeout=timeout_seconds)
    response.raise_for_status()
    return response.json()


def parse_event_occurrences(data: Any) -> list[EventOccurrence]:
    events = _extract_event_list(data)
    occurrences: list[EventOccurrence] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        occurrences.extend(_event_to_occurrences(event))
    return occurrences


def _extract_event_list(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if not isinstance(data, dict):
        return []
    for key in ("events", "data", "items", "results"):
        value = data.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _event_to_occurrences(event: dict[str, Any]) -> list[EventOccurrence]:
    event_id = str(
        event.get("id")
        or event.get("uuid")
        or event.get("slug")
        or event.get("url")
        or event.get("title")
        or "unknown-event"
    )
    title = str(event.get("title") or event.get("name") or event_id)
    description = str(
        event.get("description")
        or event.get("details")
        or event.get("body")
        or event.get("content")
        or ""
    )
    location = _coerce_location(event.get("location"))
    source_url = _coerce_str(event.get("url") or event.get("link"))
    tags = _coerce_tags(event.get("tags") or event.get("categories"))
    updated = parse_datetime(
        event.get("updated")
        or event.get("modified")
        or event.get("last_modified")
        or event.get("timestamp")
    )

    nested_occurrences = event.get("occurrences")
    if isinstance(nested_occurrences, list):
        results: list[EventOccurrence] = []
        for idx, occ in enumerate(nested_occurrences):
            if not isinstance(occ, dict):
                continue
            start = parse_datetime(
                occ.get("start") or occ.get("start_time") or occ.get("date")
            )
            if start is None:
                continue
            end = parse_datetime(occ.get("end") or occ.get("end_time"))
            occ_id = str(occ.get("id") or f"{event_id}:{idx}:{start.isoformat()}")
            results.append(
                EventOccurrence(
                    occurrence_id=occ_id,
                    event_id=event_id,
                    title=title,
                    description=description,
                    start_time=start,
                    end_time=end,
                    location=location,
                    source_url=source_url,
                    tags=tags,
                    source_type="json",
                    last_modified=updated,
                )
            )
        if results:
            return results

    start = parse_datetime(
        event.get("start")
        or event.get("start_time")
        or event.get("startDate")
        or event.get("date")
    )
    if start is None:
        return []
    end = parse_datetime(
        event.get("end") or event.get("end_time") or event.get("endDate")
    )
    occurrence_id = f"{event_id}:{start.isoformat()}"
    return [
        EventOccurrence(
            occurrence_id=occurrence_id,
            event_id=event_id,
            title=title,
            description=description,
            start_time=start,
            end_time=end,
            location=location,
            source_url=source_url,
            tags=tags,
            source_type="json",
            last_modified=updated,
        )
    ]


def _coerce_location(location: Any) -> str | None:
    if isinstance(location, str):
        return location
    if isinstance(location, dict):
        return _coerce_str(location.get("name") or location.get("address"))
    return None


def _coerce_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _coerce_tags(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value if v is not None]
    if value is None:
        return []
    return [str(value)]

