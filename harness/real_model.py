"""Provider adapters for the real-model path.

The frozen ``RealModel`` already speaks OpenAI-compatible Chat Completions.
Gemini thinking models may spend an entire completion budget on internal
reasoning and return a message without ``content``.  Requesting low reasoning
keeps the wire format unchanged while leaving room for the ReAct action/final.
"""

from __future__ import annotations

import socket
import urllib.error

from arena.model import RealModel


class RetryingRealModel(RealModel):
    """Retry one transport timeout inside the same logical model call."""

    max_transport_attempts = 2

    def _post(self, payload: dict) -> dict:
        for attempt in range(self.max_transport_attempts):
            try:
                return super()._post(payload)
            except (TimeoutError, socket.timeout):
                if attempt + 1 >= self.max_transport_attempts:
                    raise
            except urllib.error.URLError as exc:
                if not isinstance(exc.reason, (TimeoutError, socket.timeout)):
                    raise
                if attempt + 1 >= self.max_transport_attempts:
                    raise
        raise AssertionError("unreachable")


class GeminiModel(RetryingRealModel):
    """OpenAI-compatible Gemini client with bounded reasoning effort."""

    def _post(self, payload: dict) -> dict:
        payload = dict(payload)
        payload.setdefault("reasoning_effort", "low")
        return super()._post(payload)


__all__ = ["GeminiModel", "RetryingRealModel"]
