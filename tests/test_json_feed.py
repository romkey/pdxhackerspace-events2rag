from events2rag.json_feed import parse_event_occurrences


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

