from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def pick_device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_dtype(device: torch.device) -> torch.dtype:
    return torch.float32 if device.type == "cpu" else torch.float16


@dataclass
class LoadedModel:
    model: torch.nn.Module
    tokenizer: object
    device: torch.device
    dtype: torch.dtype
    pad_id: int
    eos_ids: frozenset[int]


def eos_ids_for(model, tokenizer) -> frozenset[int]:
    ids = set()
    gen = getattr(model, "generation_config", None)
    raw = getattr(gen, "eos_token_id", None) if gen is not None else None
    if isinstance(raw, int):
        ids.add(raw)
    elif raw:
        ids.update(raw)
    if tokenizer is not None and tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)
    return frozenset(ids)


def load(name: str, device: str | None = None) -> LoadedModel:
    dev = pick_device(device)
    dtype = pick_dtype(dev)
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, attn_implementation="sdpa")
    model.to(dev).eval()
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    return LoadedModel(model, tok, dev, dtype, pad, eos_ids_for(model, tok))


def kv_bytes_per_token(config, dtype: torch.dtype) -> int:
    heads = config.num_attention_heads
    kv_heads = getattr(config, "num_key_value_heads", None) or heads
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // heads
    elt = torch.tensor([], dtype=dtype).element_size()
    return 2 * config.num_hidden_layers * kv_heads * head_dim * elt


def activation_bytes_per_token(config, dtype: torch.dtype) -> int:
    elt = torch.tensor([], dtype=dtype).element_size()
    per_token = 6 * config.hidden_size + 3 * config.intermediate_size
    return 2 * per_token * elt


def attention_scratch_bytes_per_token(config, dtype: torch.dtype) -> int:
    heads = config.num_attention_heads
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // heads
    elt = torch.tensor([], dtype=dtype).element_size()
    return 2 * heads * head_dim * elt
