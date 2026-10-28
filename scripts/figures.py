"""README figure: where one inference's time goes, per configuration.

    python scripts/figures.py   ->  docs/figures/latency_breakdown.png

Every bar splits the end-to-end Policy.infer time (median over 3 runs of the per-run
p50, batch 1) the same way: the per-step denoising cost is (T10 - T1) / 9 from runs
at 10 and 1 denoising steps of the same configuration, and the rest (prefix: SigLIP +
Gemma, host transforms, copies, launch overhead) is T10 minus 10 steps. For stock
compiled this per-step cost includes the host round trip that the graph break forces
between steps. The 5-step rows use the per-step cost of their 10-step configuration.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "docs/figures"
F = "prune+static+cond+batchimg"
SERIES = ["#2a78d6", "#eb6834"]
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"


def runs(path_fmt):
    return [json.load(open(path_fmt.format(i))) for i in (1, 2, 3)]


def e2e(path_fmt):
    """Median over the three runs of the per-run end-to-end p50."""
    return sorted(r["summary"]["cpu_ms/infer_e2e"]["p50"] for r in runs(path_fmt))[1]


def rows():
    r2, st, t2 = "results/latency", "results/latency_steps", "results/latency_int8"
    out = []
    for label, t10f, t1f, tradeoff in [
        ("openpi, eager", f"{r2}/stock_eager_run{{}}.json", f"{st}/stock_eager_steps1_run{{}}.json", False),
        ("openpi, torch.compile (default)", f"{r2}/stock_compiled_run{{}}.json", f"{st}/stock_compiled_steps1_run{{}}.json", False),
        ("tier 1 (ours)", f"{r2}/{F}_compiled_run{{}}.json", f"{st}/{F}_compiled_steps1_run{{}}.json", False),
        ("+ INT8 language model", f"{t2}/{F}+int8_compiled_run{{}}.json", f"{t2}/{F}+int8_compiled_steps1_run{{}}.json", True),
    ]:
        t10, t1 = e2e(t10f), e2e(t1f)
        out.append((label, t10, 10 * (t10 - t1) / 9, tradeoff))
    step_t1, step_q = out[2][2] / 10, out[3][2] / 10
    out.append(("+ 5 denoising steps", e2e(f"{st}/{F}_compiled_steps5_run{{}}.json"), 5 * step_t1, True))
    out.append(("+ INT8 + 5 steps", e2e(f"{t2}/{F}+int8_compiled_steps5_run{{}}.json"), 5 * step_q, True))
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    data = rows()
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
                         "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK})
    fig, ax = plt.subplots(figsize=(9.5, 4.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    names = ["Everything else: prefix, host, launches", "Denoising: (T10 - T1) / 9 per step"]
    labels = []
    for y, (label, total, den, tradeoff) in enumerate(reversed(data)):
        left = 0.0
        for i, w in enumerate((total - den, den)):
            ax.barh(y, w, left=left, height=0.56, color=SERIES[i], edgecolor=SURFACE, linewidth=2,
                    hatch="//" if tradeoff else None, label=names[i] if y == len(data) - 1 else None)
            if w > 4:
                box = {"boxstyle": "round,pad=0.15", "fc": SURFACE, "ec": "none", "alpha": 0.85} if tradeoff else None
                ax.text(left + w / 2, y, f"{w:.1f}", ha="center", va="center", fontsize=9, color=INK, bbox=box)
            left += w
        ax.text(total + 1.2, y, f"{total:.1f} ms", va="center", fontsize=9.5, color=INK)
        labels.append(label)
    ax.set_yticks(range(len(data)), labels)
    ax.tick_params(axis="y", length=0)
    ax.axhline(len(data) - 3.5, color=INK_2, linewidth=0.8, linestyle=":")  # tier 1 above, tier 2 below
    ax.set_xlabel("ms per inference (median of per-run p50), RTX 5090, batch 1")
    ax.set_xlim(0, max(d[1] for d in data) * 1.12)
    fig.suptitle("pi0.5 LIBERO inference: tier 1 (same computation), then tier 2 trade-offs", x=0.01, y=0.98,
                 ha="left", fontsize=12, color=INK)
    fig.legend(*ax.get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(0.005, 0.91), ncol=2,
               frameon=False, fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(f"{OUT}/latency_breakdown.png", dpi=150, facecolor=SURFACE)
    for label, total, den, tradeoff in data:
        print(f"{label:36s} total {total:6.2f}  denoise {den:6.2f}  rest {total - den:6.2f}{'  (trade-off)' if tradeoff else ''}")


if __name__ == "__main__":
    main()
