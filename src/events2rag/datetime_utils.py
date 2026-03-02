from __future__ import annotations

from datetime import UTC, datetime

from dateutil import parser as date_parser


def parse_datetime(value: str | datetime | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return ensure_utc(value)
    if isinstance(value, str):
        parsed = date_parser.parse(value)
        return ensure_utc(parsed)
    return None


def ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)

