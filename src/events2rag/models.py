from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from events2rag.text_utils import human_date


@dataclass(frozen=True)
class EventOccurrence:
    occurrence_id: str
    event_id: str
    title: str
    description: str
    start_time: datetime
    end_time: datetime | None
    location: str | None
    source_url: str | None
    tags: list[str] = field(default_factory=list)
    source_type: str = "json"
    last_modified: datetime | None = None
    temporal_status: str = "future"
    duration: str | None = None

    def embedding_text(self) -> str:
        when = human_date(self.start_time)
        where = self.location or "unknown"
        tag_text = ", ".join(self.tags) if self.tags else "none"
        description = self.description or "no description"
        duration_text = self.duration or "unknown"
        parts = [
            f"Title: {self.title}",
            f"When: {when}",
            f"Duration: {duration_text}",
            f"Where: {where}",
            f"Tags: {tag_text}",
            f"Description: {description}",
        ]
        return "\n".join(parts)


@dataclass(frozen=True)
class EventSummary:
    event_id: str
    title: str
    description: str
    next_start_time: datetime | None
    locations: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    source_url: str | None = None
    source_type: str = "json"
    occurrence_count: int = 0
    last_modified: datetime | None = None
    frequency: str = "unknown"
    has_future_occurrences: bool = False

    def embedding_text(self) -> str:
        next_when = (
            human_date(self.next_start_time)
            if self.next_start_time
            else "unknown"
        )
        where = ", ".join(self.locations) if self.locations else "unknown"
        tag_text = ", ".join(self.tags) if self.tags else "none"
        description = self.description or "no description"
        schedule = (
            f"{self.frequency.capitalize()} "
            f"({self.occurrence_count} occurrences)"
        )
        parts = [
            f"Event: {self.title}",
            f"Schedule: {schedule}",
            f"Next Occurrence: {next_when}",
            f"Locations: {where}",
            f"Tags: {tag_text}",
            f"Description: {description}",
        ]
        return "\n".join(parts)
