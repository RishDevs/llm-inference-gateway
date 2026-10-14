from __future__ import annotations

import random
from dataclasses import dataclass

CORPUS = """
The river town woke slowly. Fishing boats came back before the market opened, and the smell of bread
drifted from the corner bakery where the same three people had worked for twenty years. Children ran
along the stone wall by the water while their parents argued about the price of tomatoes and whether
the bridge would ever be repaired. Nobody hurried, because nothing in the town had ever needed to hurry.

A distributed system is a set of computers that cooperate to look like one. The hard parts are not the
happy paths. They are partial failures, clocks that disagree, messages that arrive twice or never, and
operators who restart the wrong machine at the worst time. Good designs make the common case fast and
the failure case boring, and they write down exactly what is guaranteed and what is not.

To make a simple tomato soup, soften an onion and two cloves of garlic in olive oil, add a tin of
tomatoes and a cup of stock, simmer for twenty minutes, then blend until smooth. Season with salt,
pepper and a little sugar if the tomatoes are sharp. A spoon of cream at the end is optional but nice.

The committee met on Thursday to review the budget for the coming year. Most of the discussion was
about maintenance: roofs that leak, buses that need new engines, software licenses that renew every
spring. The new library wing was approved after a long debate, on the condition that construction
starts after the school term ends so the noise does not disturb exams.

Training for a first marathon usually takes sixteen to twenty weeks. Most of the running should be
easy enough to hold a conversation. One long run a week builds endurance, and it grows by a mile or two
at a time, with a lighter week every month so the body can recover. Sleep matters as much as mileage.
"""

PROMPT_BUCKETS = (64, 128, 256, 512)
OUTPUT_BUCKETS = (32, 64, 128, 256)


@dataclass(frozen=True)
class Job:
    prompt_ids: list[int]
    max_tokens: int


def corpus_ids(tokenizer) -> list[int]:
    ids = tokenizer.encode(CORPUS.strip())
    while len(ids) < 4096:
        ids = ids + ids
    return ids


def jitter(rng: random.Random, top: int) -> int:
    return rng.randint(top // 2 + 1, top)


def mixed_jobs(
    tokenizer,
    count: int,
    seed: int = 0,
    prompt_buckets=PROMPT_BUCKETS,
    output_buckets=OUTPUT_BUCKETS,
) -> list[Job]:
    rng = random.Random(seed)
    source = corpus_ids(tokenizer)
    jobs = []
    for _ in range(count):
        plen = jitter(rng, rng.choice(prompt_buckets))
        out = jitter(rng, rng.choice(output_buckets))
        start = rng.randrange(0, len(source) - plen)
        jobs.append(Job(source[start:start + plen], out))
    return jobs
