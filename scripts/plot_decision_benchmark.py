#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Plot the Markdown output of scripts.benchmark_decision_ane as PNG and SVG.

Run: uv run --no-project --with matplotlib python -m scripts.plot_decision_benchmark RESULTS.md
The figures are saved beside the input; project dependencies are unchanged.
"""

import argparse
import math
from pathlib import Path

MODES = ("GPU", "Single ANE + GPU")
CASES = ("short", "long")


def read_rows(source: Path) -> dict:
    rows = {}
    for line in source.read_text().splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 6 or cells[0] not in MODES:
            continue
        key = (cells[0], cells[1])
        if key in rows:
            raise ValueError(f"Duplicate benchmark row: {key}")
        tokens, seconds, memory, calls = (
            int(cells[2]),
            float(cells[3]),
            float(cells[4]),
            int(cells[5]),
        )
        if (
            tokens <= 0
            or calls < 0
            or not math.isfinite(seconds)
            or seconds <= 0
            or not math.isfinite(memory)
            or memory < 0
        ):
            raise ValueError(f"Invalid benchmark measurements: {key}")
        rows[key] = (tokens, seconds, memory, calls)
    if set(rows) != {(mode, case) for mode in MODES for case in CASES}:
        raise ValueError("Expected GPU and ANE rows for both short and long cases")
    for case in CASES:
        if rows[(MODES[0], case)][0] != rows[(MODES[1], case)][0]:
            raise ValueError("Cannot compare different token counts")
    return rows


def plot(source: Path) -> None:
    rows = read_rows(source)
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    plt.rcParams.update(
        {"font.size": 11, "axes.spines.top": False, "axes.spines.right": False}
    )
    metrics = {
        (mode, case): (
            seconds,
            tokens / seconds,
            rows[(MODES[0], case)][1] / seconds,
        )
        for (mode, case), (tokens, seconds, _, _) in rows.items()
    }
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.0))
    colors = ("#4169A6", "#408A70")
    for ax, column, title, unit in zip(
        axes,
        (0, 1, 2),
        ("Median body time", "Prompt throughput", "Versus GPU"),
        (
            "Seconds · lower is better",
            "Tokens/s · higher is better",
            "Speedup · higher is better",
        ),
    ):
        for i, mode in enumerate(MODES):
            values = [metrics[(mode, case)][column] for case in CASES]
            bars = ax.bar(
                [x + (i - 0.5) * 0.36 for x in range(2)],
                values,
                width=0.34,
                label=mode,
                color=colors[i],
            )
            ax.bar_label(
                bars,
                labels=[f"{v:.1f}" if column == 1 else f"{v:.3f}" for v in values],
                padding=4,
                fontsize=10,
            )
        ax.set_xticks(
            range(2),
            [f"{case.title()}\n{rows[(MODES[0], case)][0]:,} tokens" for case in CASES],
        )
        ax.set_ylim(0, max(row[column] for row in metrics.values()) * 1.25)
        if column == 2:
            ax.axhline(1, color="#555555", linestyle="--", linewidth=1)
        ax.set_ylabel(unit)
        ax.set_title(title, pad=14)
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)

    fig.suptitle("Clef-Flash oQ4 · GPU/ANE benchmark summary", fontsize=15, y=0.98)
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=2,
        frameon=False,
    )
    fig.text(
        0.5,
        0.035,
        f"Native ANE MLP calls (all runs): short = {rows[(MODES[1], 'short')][3]}, "
        f"long = {rows[(MODES[1], 'long')][3]}.\n"
        "Body time includes backbone, decision head and executor scheduling; excludes load, encoding and HTTP.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.12, 1, 0.86), w_pad=2)
    for suffix in (".png", ".svg"):
        fig.savefig(source.with_suffix(suffix), dpi=180, facecolor="white")
    plt.close(fig)
    print("Saved PNG and SVG next to the input Markdown file.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    plot(parser.parse_args().results)


if __name__ == "__main__":
    main()
