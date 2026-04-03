from __future__ import annotations


def sort_by_time(items):
    return sorted(items, key=lambda x: (x.start, x.end))


def validate_monotonic(items, label: str) -> None:
    for item in items:
        if item.start < 0:
            raise ValueError(f"{label}: start negativo detectado")
        if item.end < 0:
            raise ValueError(f"{label}: end negativo detectado")
        if item.start > item.end:
            raise ValueError(f"{label}: start > end detectado")
