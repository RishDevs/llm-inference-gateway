from __future__ import annotations

import os
from dataclasses import dataclass, fields, replace

PRESETS = {
    "gateway": {},
    "unbucketed": {"bucketing": False},
    "naive": {
        "max_batch_size": 1,
        "bucketing": False,
        "admission": False,
        "max_wait_ms": 0.0,
        "join_wait_ms": 0.0,
    },
    "unguarded": {
        "admission": False,
        "max_batch_size": 256,
    },
}


@dataclass(frozen=True)
class Settings:
    model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    served_name: str | None = None
    host: str = "0.0.0.0"
    port: int = 8000
    device: str | None = None
    max_model_len: int = 2048
    max_batch_size: int = 32
    max_lanes: int = 3
    lane_split_factor: float = 1.5
    rebalance_every: int = 8
    max_wait_ms: float = 30.0
    join_wait_ms: float = 15.0
    max_prefill_tokens: int = 8192
    token_budget: int = 0
    kv_memory_fraction: float = 0.85
    max_queue: int = 48
    queue_timeout_s: float = 30.0
    retry_after_s: int = 1
    bucketing: bool = True
    admission: bool = True
    preset: str = "gateway"

    @property
    def name(self) -> str:
        return self.served_name or self.model

    def validate(self) -> "Settings":
        if self.max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        if self.admission and self.max_prefill_tokens < self.max_model_len:
            raise ValueError("max_prefill_tokens must be >= max_model_len")
        return self


def _coerce(kind: str, raw: str):
    if kind == "bool":
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if kind == "int":
        return int(raw)
    if kind == "float":
        return float(raw)
    if kind == "str | None":
        return raw or None
    return raw


def from_env(env=None) -> Settings:
    env = os.environ if env is None else env
    preset = env.get("GATEWAY_PRESET", "gateway")
    base = apply_preset(Settings(), preset)
    found = {}
    for f in fields(Settings):
        key = f"GATEWAY_{f.name.upper()}"
        if key in env and f.name != "preset":
            found[f.name] = _coerce(f.type, env[key])
    return replace(base, **found)


def apply_preset(settings: Settings, preset: str) -> Settings:
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}, pick one of {sorted(PRESETS)}")
    return replace(settings, preset=preset, **PRESETS[preset])
