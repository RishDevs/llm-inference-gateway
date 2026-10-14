from __future__ import annotations

import threading

import torch

from .model import activation_bytes_per_token, attention_scratch_bytes_per_token, kv_bytes_per_token


class QueueFull(Exception):
    def __init__(self, retry_after: int):
        super().__init__("queue is full")
        self.retry_after = retry_after


class RequestTooLarge(Exception):
    pass


class TokenBudget:
    def __init__(self, capacity: int | None):
        self.capacity = capacity
        self.used = 0
        self._lock = threading.Lock()

    @property
    def unlimited(self) -> bool:
        return self.capacity is None

    def fits(self, tokens: int) -> bool:
        if self.capacity is None:
            return True
        return self.used + tokens <= self.capacity

    def take(self, tokens: int) -> bool:
        with self._lock:
            if not self.fits(tokens):
                return False
            self.used += tokens
            return True

    def give_back(self, tokens: int) -> None:
        with self._lock:
            self.used = max(0, self.used - tokens)

    @property
    def free(self) -> int | None:
        return None if self.capacity is None else self.capacity - self.used


def size_budget(settings, config, dtype, device) -> int | None:
    if not settings.admission:
        return None
    if settings.token_budget > 0:
        return settings.token_budget
    floor = settings.max_model_len
    device = torch.device(device)
    if device.type != "cuda":
        return max(floor, settings.max_model_len * settings.max_batch_size)
    free, _ = torch.cuda.mem_get_info(device)
    headroom = settings.max_prefill_tokens * activation_bytes_per_token(config, dtype)
    usable = free * settings.kv_memory_fraction - headroom
    per_token = kv_bytes_per_token(config, dtype) + attention_scratch_bytes_per_token(config, dtype)
    return max(floor, int(usable // per_token))
