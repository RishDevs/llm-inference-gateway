import pytest
import torch
from transformers import AutoModelForCausalLM, Qwen2Config

from gateway.config import Settings
from gateway.engine import ModelRunner

VOCAB = 300
PAD = 0
EOS = 1


class ByteTokenizer:
    eos_token_id = EOS
    pad_token_id = PAD

    def encode(self, text):
        return [b + 2 for b in text.encode("utf-8")]

    def decode(self, ids, skip_special_tokens=True):
        raw = bytes(i - 2 for i in ids if 2 <= i < 258)
        return raw.decode("utf-8", errors="replace")


@pytest.fixture(scope="session")
def tiny_model():
    torch.manual_seed(0)
    conf = Qwen2Config(
        vocab_size=VOCAB,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=1024,
        eos_token_id=EOS,
        pad_token_id=PAD,
    )
    model = AutoModelForCausalLM.from_config(conf, attn_implementation="sdpa")
    return model.eval()


@pytest.fixture
def runner(tiny_model):
    return ModelRunner(tiny_model, PAD, {EOS}, "cpu", torch.float32)


@pytest.fixture
def tokenizer():
    return ByteTokenizer()


@pytest.fixture
def settings():
    return Settings(model="tiny", device="cpu", max_model_len=512, max_prefill_tokens=2048, max_wait_ms=5.0)
