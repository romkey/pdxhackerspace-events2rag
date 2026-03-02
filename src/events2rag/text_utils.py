from __future__ import annotations

import re
from datetime import datetime, timedelta

from bs4 import BeautifulSoup


def strip_html(text: str) -> str:
    if not text or "<" not in text:
        return text
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text(separator=" ", strip=True)


def human_date(dt: datetime) -> str:
    return dt.strftime("%A, %B %-d, %Y at %-I:%M %p")


def human_duration(
    start: datetime, end: datetime | None
) -> str | None:
    if end is None:
        return None
    delta = end - start
    if delta <= timedelta(0):
        return None
    total_minutes = int(delta.total_seconds() // 60)
    if total_minutes < 60:
        return f"{total_minutes} minutes"
    hours, minutes = divmod(total_minutes, 60)
    if hours >= 24:
        days = hours // 24
        remaining_hours = hours % 24
        if remaining_hours:
            d = f"{days} day{'s' if days != 1 else ''}"
            h = f"{remaining_hours} hour{'s' if remaining_hours != 1 else ''}"
            return f"{d}, {h}"
        return f"{days} day{'s' if days != 1 else ''}"
    if minutes:
        return f"{hours} hour{'s' if hours != 1 else ''}, {minutes} minutes"
    return f"{hours} hour{'s' if hours != 1 else ''}"


def temporal_status(
    start: datetime, end: datetime | None, now: datetime
) -> str:
    effective_end = end or start
    if effective_end < now:
        return "past"
    if start <= now <= effective_end:
        return "current"
    return "future"


def estimate_frequency(
    start_times: list[datetime],
) -> str:
    if len(start_times) < 2:
        return "one-time"
    sorted_times = sorted(start_times)
    gaps_days = [
        (sorted_times[i + 1] - sorted_times[i]).days
        for i in range(len(sorted_times) - 1)
    ]
    avg_gap = sum(gaps_days) / len(gaps_days)
    if avg_gap <= 1.5:
        return "daily"
    if avg_gap <= 9:
        return "weekly"
    if avg_gap <= 18:
        return "biweekly"
    if avg_gap <= 45:
        return "monthly"
    return "occasional"


_WHITESPACE_RE = re.compile(r"\s+")


def collapse_whitespace(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()
