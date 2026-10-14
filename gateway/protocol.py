from __future__ import annotations

import json
import time

from pydantic import BaseModel, Field, field_validator


class StreamOptions(BaseModel):
    include_usage: bool = False


class CompletionRequest(BaseModel):
    model: str | None = None
    prompt: str | list[int]
    max_tokens: int = Field(default=128, ge=1)
    temperature: float = Field(default=1.0, ge=0.0, le=5.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    stream: bool = False
    stop: str | list[str] | None = None
    ignore_eos: bool = False
    stream_options: StreamOptions | None = None

    @field_validator("prompt")
    @classmethod
    def _not_empty(cls, v):
        if len(v) == 0:
            raise ValueError("prompt must not be empty")
        return v

    def stop_list(self) -> list[str]:
        if self.stop is None:
            return []
        if isinstance(self.stop, str):
            return [self.stop] if self.stop else []
        return [s for s in self.stop if s]


class TextStream:
    def __init__(self, tokenizer, stops: list[str] | None = None):
        self.tokenizer = tokenizer
        self.stops = stops or []
        self.ids: list[int] = []
        self.sent = 0
        self.text = ""
        self.stopped = False

    def feed(self, token: int) -> str:
        self.ids.append(token)
        text = self.tokenizer.decode(self.ids, skip_special_tokens=True)
        if text.endswith("�"):
            return ""
        self.text = text
        return self._flush(final=False)

    def close(self) -> str:
        if self.stopped:
            return ""
        self.text = self.tokenizer.decode(self.ids, skip_special_tokens=True).rstrip("�")
        return self._flush(final=True)

    def _flush(self, final: bool) -> str:
        if self.stopped:
            return ""
        for stop in self.stops:
            at = self.text.find(stop, max(0, self.sent - len(stop)))
            if at != -1:
                self.stopped = True
                chunk = self.text[self.sent:at]
                self.sent = at
                return chunk
        hold = 0 if final else max((len(s) - 1 for s in self.stops), default=0)
        upto = max(self.sent, len(self.text) - hold)
        chunk = self.text[self.sent:upto]
        self.sent = upto
        return chunk

    @property
    def emitted(self) -> str:
        return self.text[: self.sent]


def chunk(rid: str, model: str, text: str, finish: str | None = None, usage: dict | None = None) -> str:
    body = {
        "id": rid,
        "object": "text_completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "text": text, "logprobs": None, "finish_reason": finish}],
    }
    if usage is not None:
        body["usage"] = usage
    return f"data: {json.dumps(body)}\n\n"


def usage_chunk(rid: str, model: str, usage: dict) -> str:
    body = {
        "id": rid,
        "object": "text_completion",
        "created": int(time.time()),
        "model": model,
        "choices": [],
        "usage": usage,
    }
    return f"data: {json.dumps(body)}\n\n"


def completion(rid: str, model: str, text: str, finish: str, usage: dict) -> dict:
    return {
        "id": rid,
        "object": "text_completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "text": text, "logprobs": None, "finish_reason": finish}],
        "usage": usage,
    }


def error_body(message: str, kind: str) -> dict:
    return {"error": {"message": message, "type": kind}}
