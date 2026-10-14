from __future__ import annotations

import argparse
import json
from pathlib import Path

ORDER = ["naive", "unbucketed", "gateway", "vllm"]


def load_runs(folder: Path) -> dict:
    runs = {}
    for f in sorted(folder.glob("*.json")):
        try:
            data = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        if "label" in data:
            runs[data["label"]] = data
    return runs


def level(run: dict | None, c: int) -> dict | None:
    if not run:
        return None
    for lv in run.get("levels", []):
        if lv["concurrency"] == c:
            return lv["summary"]
    return None


def fmt(v, unit=""):
    if v is None:
        return "—"
    if unit == "ms":
        return f"{v * 1000:.0f} ms"
    if unit == "s":
        return f"{v:.2f} s"
    if unit == "x":
        return f"{v:.2f}×"
    if unit == "%":
        return f"{v * 100:.1f}%"
    return f"{v:.1f}"


def ratio(a, b):
    if a is None or b in (None, 0):
        return None
    return a / b


def get(summary, key):
    return summary.get(key) if summary else None


def steady_table(runs: dict) -> list[str]:
    labels = [l for l in ORDER if l in runs and runs[l].get("levels")]
    labels += [l for l in runs if l not in labels and runs[l].get("levels")]
    levels = sorted({lv["concurrency"] for l in labels for lv in runs[l]["levels"]})
    lines = ["| System | Concurrency | Throughput (tok/s) | TTFT p50 | TTFT p99 | E2E p50 | E2E p99 | OK / sent |",
             "|---|---|---|---|---|---|---|---|"]
    for l in labels:
        for c in levels:
            s = level(runs[l], c)
            if not s:
                continue
            lines.append(
                f"| {l} | {c} | {fmt(get(s, 'throughput_tok_s'))} | {fmt(get(s, 'ttft_p50_s'), 'ms')} | "
                f"{fmt(get(s, 'ttft_p99_s'), 'ms')} | {fmt(get(s, 'e2e_p50_s'), 's')} | {fmt(get(s, 'e2e_p99_s'), 's')} | "
                f"{s.get('completed', 0)} / {s.get('requests', 0)} |"
            )
    return lines


def headline(runs: dict, profile: dict | None, c: int) -> list[str]:
    naive, unb, gw, vllm = (level(runs.get(k), c) for k in ORDER)
    rows = [
        ("Throughput, micro-batching (unbucketed) vs naive", ratio(get(unb, "throughput_tok_s"), get(naive, "throughput_tok_s")), "x"),
        ("Throughput, length-bucketed vs unbucketed", ratio(get(gw, "throughput_tok_s"), get(unb, "throughput_tok_s")), "x"),
        ("Throughput, full gateway vs naive", ratio(get(gw, "throughput_tok_s"), get(naive, "throughput_tok_s")), "x"),
        ("Gateway throughput as a share of vLLM", ratio(get(gw, "throughput_tok_s"), get(vllm, "throughput_tok_s")), "%"),
        ("TTFT p50, naive", get(naive, "ttft_p50_s"), "ms"),
        ("TTFT p50, gateway", get(gw, "ttft_p50_s"), "ms"),
        ("E2E p99 reduction, gateway vs naive",
         None if not (gw and naive) else 1 - gw["e2e_p99_s"] / naive["e2e_p99_s"], "%"),
    ]
    if profile:
        f, b = profile["fifo"], profile["bucketed"]
        rows += [
            ("KV padding share of decode time, FIFO (profiler)", f["kv_padding_share_of_decode_time"], "%"),
            ("All padding share of engine time, FIFO (profiler)", f["padding_share_of_engine_time"], "%"),
            ("All padding share of engine time, bucketed (profiler)", b["padding_share_of_engine_time"], "%"),
            ("Padded tokens cut by bucketing (profiler)", profile["padded_token_reduction"], "%"),
            ("Throughput, bucketed vs FIFO (profiler, in-process)", profile["throughput_ratio"], "x"),
        ]
    out = [f"| Metric (concurrency {c}) | Value |", "|---|---|"]
    out += [f"| {name} | {fmt(v, unit)} |" for name, v, unit in rows]
    return out


def burst_table(runs: dict) -> list[str]:
    lines = ["| Run | Phase | Sent | OK | Status counts | TTFT p50 | Throughput (tok/s) | OOMs (server) | Peak GPU mem |",
             "|---|---|---|---|---|---|---|---|---|"]
    found = False
    for label, run in runs.items():
        if run.get("mode") != "burst":
            continue
        found = True
        stats = run.get("server_stats") or {}
        ooms = stats.get("ooms", "—")
        peak = stats.get("gpu_peak_bytes")
        peak_s = f"{peak / 2**30:.2f} GiB" if peak else "—"
        for phase, s in run["burst"]["by_phase"].items():
            lines.append(
                f"| {label} | {phase} | {s.get('requests', 0)} | {s.get('completed', 0)} | {s.get('statuses')} | "
                f"{fmt(get(s, 'ttft_p50_s'), 'ms')} | {fmt(get(s, 'throughput_tok_s'))} | {ooms} | {peak_s} |"
            )
    return lines if found else []


def plots(runs: dict, out: Path) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    made = []
    steady = {l: r for l, r in runs.items() if r.get("levels")}
    for key, title, fname, scale in [
        ("throughput_tok_s", "Output throughput (tokens/s)", "throughput.png", 1),
        ("ttft_p50_s", "Time to first token, p50 (ms)", "ttft_p50.png", 1000),
        ("e2e_p99_s", "End-to-end latency, p99 (s)", "e2e_p99.png", 1),
    ]:
        fig, ax = plt.subplots(figsize=(6, 4))
        for label, run in steady.items():
            xs = [lv["concurrency"] for lv in run["levels"] if lv["summary"].get(key) is not None]
            ys = [lv["summary"][key] * scale for lv in run["levels"] if lv["summary"].get(key) is not None]
            ax.plot(xs, ys, marker="o", label=label)
        ax.set_xlabel("concurrent users")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / fname, dpi=130)
        plt.close(fig)
        made.append(fname)
    return made


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Summarize benchmark JSON files into a markdown report")
    p.add_argument("folder", nargs="?", default="results")
    p.add_argument("--concurrency", type=int, default=32)
    args = p.parse_args(argv)
    folder = Path(args.folder)
    runs = load_runs(folder)
    prof_path = folder / "profile" / "profile.json"
    profile = json.loads(prof_path.read_text()) if prof_path.exists() else None

    md = ["# Benchmark summary", ""]
    any_run = next(iter(runs.values()), None)
    if any_run:
        md += [f"Model: `{any_run['model']}`", ""]
    md += ["## Headline", ""] + headline(runs, profile, args.concurrency) + [""]
    md += ["## Steady-state sweep", ""] + steady_table(runs) + [""]
    burst = burst_table(runs)
    if burst:
        md += ["## Burst test", ""] + burst + [""]
    pics = plots(runs, folder)
    if pics:
        md += ["## Plots", ""] + [f"![{p}]({p})" for p in pics] + [""]
    (folder / "summary.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
