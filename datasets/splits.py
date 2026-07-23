"""Read and validate publication split manifests."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Mapping, Sequence


def load_event_ids(path: str | Path) -> List[str]:
    """Load non-empty IDs, rejecting duplicates and placeholder-only files."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Split manifest does not exist: {path}")
    ids = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not ids:
        raise ValueError(f"Split manifest has no event IDs: {path}")
    duplicates = sorted({event_id for event_id in ids if ids.count(event_id) > 1})
    if duplicates:
        raise ValueError(f"Duplicate event IDs in {path}: {', '.join(duplicates)}")
    return ids


def assert_disjoint(splits: Mapping[str, Sequence[str]]) -> None:
    """Fail when the same event is assigned to more than one split."""
    owners: dict[str, str] = {}
    for split_name, event_ids in splits.items():
        for event_id in event_ids:
            if event_id in owners:
                raise ValueError(
                    f"Event {event_id!r} appears in both {owners[event_id]!r} and {split_name!r}."
                )
            owners[event_id] = split_name


def require_member(event_id: str, expected_split: Iterable[str], split_name: str) -> None:
    """Protect evaluation scripts from scoring an event outside the requested split."""
    if event_id not in set(expected_split):
        raise ValueError(f"Event {event_id!r} is not declared in the {split_name} split.")
