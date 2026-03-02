from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


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

    def embedding_text(self) -> str:
        when = self.start_time.isoformat()
        where = self.location or "unknown"
        tag_text = ", ".join(self.tags) if self.tags else "none"
        description = self.description or "no description"
        return (
            f"Title: {self.title}\n"
            f"When: {when}\n"
            f"Where: {where}\n"
            f"Tags: {tag_text}\n"
            f"Description: {description}"
        )

