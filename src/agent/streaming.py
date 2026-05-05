"""Streaming helpers for agent SSE responses."""
from __future__ import annotations

import json

from .models import StreamEvent


class StreamingAdapter:
    """Stateless SSE formatting utilities."""

    @staticmethod
    def to_sse(event: StreamEvent) -> str:
        return f"event: {event.event_type}\ndata: {json.dumps(event.data)}\n\n"

    @staticmethod
    def error_event(code: str, message: str) -> str:
        return f'event: error\ndata: {json.dumps({"code": code, "message": message})}\n\n'

    @staticmethod
    def sources_event(source_dicts: list[dict]) -> str:
        return f"event: sources\ndata: {json.dumps(source_dicts)}\n\n"

    @staticmethod
    def make_event(event_type: str, data: dict) -> StreamEvent:
        return StreamEvent(event_type=event_type, data=data)
