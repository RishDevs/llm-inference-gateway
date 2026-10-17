from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

METRICS = [
    ("throughput_tok_s", "Throughput (tok/s)", 1.0, "{:.1f}"),
    ("ttft_p50_s", "TTFT p50 (ms)", 1000.0, "{:.0f}"),
    ("e2e_p99_s", "E2E p99 (s)", 1.0, "{:.2f}"),
    ("queue_p50_s", "Server queue wait p50 (ms)", 1000.0, "{:.0f}"),
    ("prefill_p50_s", "Server prefill p50 (ms)", 1000.0, "{:.0f}"),
]


def collect(folder: Path) -> dict:
    groups: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for f in sorted(folder.rglob("*.json")):
        try:
            data = json.loads(f.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if data.get("mode") != "steady":
            continue
        name = re.sub(r"-r\d+$", "", data["label"])
        for lv in data.get("levels", []):
            s = dict(lv["summary"])
            s.update({k: v for k, v in (lv.get("server") or {}).items() if k in ("queue_p50_s", "prefill_p50_s")})
            for key, *_ in METRICS:
                if s.get(key) is not None:
                    groups[(name, lv["concurrency"])][key].append(s[key])
    return groups


def spread(values: list[float], scale: float, fmt: str) -> str:
    mean = statistics.fmean(values) * scale
    if len(values) < 2:
        return fmt.format(mean)
    sd = statistics.stdev(values) * scale
    return f"{fmt.format(mean)} ± {fmt.format(sd)}"


def mean(groups, name, c, key):
    vals = groups.get((name, c), {}).get(key)
    return statistics.fmean(vals) if vals else None


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Average repeated benchmark runs (labels ending in -rN)")
    p.add_argument("folder")
    p.add_argument("--baseline", default="naive")
    p.add_argument("--reference", default="vllm")
    args = p.parse_args(argv)
    folder = Path(args.folder)
    groups = collect(folder)

    lines = ["# Averaged results", "", "Mean ± standard deviation across repeated runs.", ""]
    lines += ["| System | Users | Runs | " + " | ".join(label for _, label, _, _ in METRICS) + " |"]
    lines += ["|---|---|---|" + "---|" * len(METRICS)]
    for (name, c), vals in sorted(groups.items()):
        runs = max(len(v) for v in vals.values())
        cells = [spread(vals[key], scale, fmt) if key in vals else "—" for key, _, scale, fmt in METRICS]
        lines.append(f"| {name} | {c} | {runs} | " + " | ".join(cells) + " |")

    ratios = []
    for name, c in sorted(groups):
        if name in (args.baseline, args.reference):
            continue
        tput = mean(groups, name, c, "throughput_tok_s")
        base = mean(groups, args.baseline, c, "throughput_tok_s")
        ref = mean(groups, args.reference, c, "throughput_tok_s")
        p99 = mean(groups, name, c, "e2e_p99_s")
        base99 = mean(groups, args.baseline, c, "e2e_p99_s")
        if base:
            ratios.append(f"- {name} vs {args.baseline} at {c} users: {tput / base:.1f}× throughput")
        if base99:
            ratios.append(f"- {name} vs {args.baseline} at {c} users: {1 - p99 / base99:.0%} lower p99 latency")
        if ref:
            ratios.append(f"- {name} as a share of {args.reference} at {c} users: {tput / ref:.1%}")
    if ratios:
        lines += ["", "## Ratios (of means)", ""] + ratios

    text = "\n".join(lines) + "\n"
    (folder / "aggregate.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
