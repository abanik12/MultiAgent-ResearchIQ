"""Context-local queue for curated agent trace events during graph runs."""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any

_trace_queue: ContextVar[list[dict[str, Any]] | None] = ContextVar("_trace_queue", default=None)


def activate_trace_queue() -> None:
    _trace_queue.set([])


def deactivate_trace_queue() -> None:
    _trace_queue.set(None)


def drain_trace_events() -> list[dict[str, Any]]:
    queue = _trace_queue.get()
    if not queue:
        return []
    events = list(queue)
    queue.clear()
    return events


def emit_trace_event(event_type: str | None = None, **payload: Any) -> None:
    queue = _trace_queue.get()
    if queue is None:
        return
    if event_type is None:
        event_type = str(payload.pop("type", "trace"))
    else:
        payload.pop("type", None)
    queue.append({"type": event_type, **payload})
