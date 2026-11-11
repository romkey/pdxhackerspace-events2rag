from datetime import UTC, datetime

import requests

from events2rag.json_feed import (
    fetch_json,
    parse_event_occurrences,
)


def test_parse_event_occurrences_with_nested_occurrences() -> None:
    payload = {
        "events": [
            {
                "id": "soldering-night",
                "title": "Soldering Night",
                "description": "Build things",
                "location": "PDX Hackerspace",
                "url": "https://example.com/soldering",
                "tags": ["electronics", "beginner"],
                "occurrences": [
                    {"id": "occ-1", "start": "2026-03-01T02:00:00Z"},
                    {"id": "occ-2", "start": "2026-03-08T02:00:00Z"},
                ],
            }
        ]
    }

    occurrences = parse_event_occurrences(payload)
    assert len(occurrences) == 2
    assert occurrences[0].event_id == "soldering-night"
    assert occurrences[0].occurrence_id == "occ-1"
    assert occurrences[0].tags == ["electronics", "beginner"]


def test_parse_event_occurrences_single_date() -> None:
    payload = {
        "events": [
            {
                "id": "open-house",
                "title": "Open House",
                "start": "2026-03-10T01:00:00Z",
            }
        ]
    }
    occurrences = parse_event_occurrences(payload)
    assert len(occurrences) == 1
    assert occurrences[0].occurrence_id.startswith("open-house:")


def test_html_stripped_from_description() -> None:
    payload = {
        "events": [
            {
                "id": "html-event",
                "title": "Event",
                "description": "<p>Join us for <b>fun</b>!</p>",
                "start": "2026-03-10T01:00:00Z",
            }
        ]
    }
    occurrences = parse_event_occurrences(payload)
    assert "<p>" not in occurrences[0].description
    assert "<b>" not in occurrences[0].description
    assert "fun" in occurrences[0].description


def test_extract_from_list_and_alternate_keys() -> None:
    event = {"id": "e", "title": "T", "start": "2026-03-10T01:00:00Z"}

    assert len(parse_event_occurrences([event, "junk"])) == 1
    for key in ("data", "items", "results"):
        assert len(parse_event_occurrences({key: [event]})) == 1


def test_unrecognized_payloads_yield_nothing() -> None:
    assert parse_event_occurrences("not json") == []
    assert parse_event_occurrences({"events": "nope"}) == []
    assert parse_event_occurrences({"other": []}) == []


def test_non_dict_events_are_skipped() -> None:
    event = {"id": "e", "title": "T", "start": "2026-03-10T01:00:00Z"}
    occurrences = parse_event_occurrences({"events": [None, 3, event]})
    assert [occ.event_id for occ in occurrences] == ["e"]


def test_event_without_start_is_dropped() -> None:
    assert parse_event_occurrences({"events": [{"id": "no-date"}]}) == []


def test_alternate_field_names_and_coercions() -> None:
    payload = [
        {
            "slug": "repair-cafe",
            "name": "Repair Cafe",
            "details": "Fix things",
            "location": {"address": "123 Main St"},
            "link": "https://example.com/repair",
            "categories": "community",
            "modified": "2026-01-02T00:00:00Z",
            "startDate": "2026-03-10T01:00:00Z",
            "endDate": "2026-03-10T03:00:00Z",
        }
    ]

    occ = parse_event_occurrences(payload)[0]

    assert occ.event_id == "repair-cafe"
    assert occ.title == "Repair Cafe"
    assert occ.description == "Fix things"
    assert occ.location == "123 Main St"
    assert occ.source_url == "https://example.com/repair"
    assert occ.tags == ["community"]
    assert occ.last_modified == datetime(2026, 1, 2, tzinfo=UTC)
    assert occ.end_time == datetime(2026, 3, 10, 3, tzinfo=UTC)


def test_defaults_when_fields_missing() -> None:
    occ = parse_event_occurrences(
        [{"date": "2026-03-10", "location": 42, "tags": ["a", None]}]
    )[0]

    assert occ.event_id == "unknown-event"
    assert occ.title == "unknown-event"
    assert occ.description == ""
    assert occ.location is None
    assert occ.source_url is None
    assert occ.tags == ["a"]
    assert occ.last_modified is None


def test_nested_occurrences_skip_invalid_and_generate_ids() -> None:
    payload = {
        "events": [
            {
                "id": "e",
                "title": "T",
                "occurrences": [
                    "junk",
                    {"start": None},
                    {
                        "start_time": "2026-03-01T02:00:00Z",
                        "end_time": "2026-03-01T04:00:00Z",
                    },
                ],
            }
        ]
    }

    occurrences = parse_event_occurrences(payload)

    assert len(occurrences) == 1
    assert occurrences[0].occurrence_id == "e:2:2026-03-01T02:00:00+00:00"
    assert occurrences[0].end_time == datetime(2026, 3, 1, 4, tzinfo=UTC)


def test_nested_occurrences_all_invalid_falls_back_to_top_level() -> None:
    payload = [
        {
            "id": "e",
            "title": "T",
            "start": "2026-03-10T01:00:00Z",
            "occurrences": [{"start": None}],
        }
    ]

    occurrences = parse_event_occurrences(payload)

    assert len(occurrences) == 1
    assert occurrences[0].occurrence_id == "e:2026-03-10T01:00:00+00:00"


def test_fetch_json(monkeypatch) -> None:
    seen: dict = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            seen["raised"] = True

        def json(self):
            return {"events": []}

    def fake_get(url, timeout=None):
        seen["url"] = url
        seen["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(requests, "get", fake_get)

    assert fetch_json("https://example.com/e.json", 3) == {"events": []}
    assert seen == {
        "url": "https://example.com/e.json",
        "timeout": 3,
        "raised": True,
    }
