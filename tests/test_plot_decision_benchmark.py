# SPDX-License-Identifier: Apache-2.0
"""Validate plotting inputs without requiring Matplotlib or a GPU."""

import pytest

from scripts.benchmark_decision_ane import print_summary
from scripts.plot_decision_benchmark import read_rows

TABLE = """Recorded results, followed by the benchmark's normal table.
| Mode | Case | Tokens | Warm median (s) | Process GiB | ANE MLP operations |
|---|---|---:|---:|---:|---:|
| GPU | short | 361 | 1.759 | 6.89 | 0 |
| GPU | long | 1963 | 8.470 | 7.71 | 0 |
| Single ANE + GPU | short | 361 | 1.706 | 7.57 | 0 |
| Single ANE + GPU | long | 1963 | 8.189 | 8.29 | 96 |

Smoke checks passed.
"""


def test_reads_table_among_other_output(tmp_path):
    source = tmp_path / "results.md"
    source.write_text(TABLE)
    rows = read_rows(source)
    assert len(rows) == 4
    assert rows[("Single ANE + GPU", "long")] == (1963, 8.189, 8.29, 96)


def test_summary_reports_per_case_throughput_and_speedup(tmp_path, capsys):
    source = tmp_path / "results.md"
    source.write_text(TABLE)
    rows = read_rows(source)
    print_summary(rows)
    output = capsys.readouterr().out
    assert (
        output.count("| Path | Median body time | Prompt throughput | Versus GPU |")
        == 2
    )
    assert "| GPU | 8.4700 s | 231.8 tok/s | 1.000x |" in output
    assert "| Single ANE + GPU | 8.1890 s | 239.7 tok/s | 1.034x |" in output
    assert "| Single ANE + GPU | 1.7060 s | 211.6 tok/s | 1.031x |" in output
    source.write_text(output)
    assert read_rows(source) == rows


def test_summary_rejects_different_token_counts(tmp_path):
    source = tmp_path / "results.md"
    source.write_text(TABLE)
    rows = read_rows(source)
    rows[("Single ANE + GPU", "long")] = (2048, 8.189, 8.29, 96)
    with pytest.raises(ValueError, match="different token counts"):
        print_summary(rows)


@pytest.mark.parametrize(
    "text",
    [
        TABLE.replace("| GPU | short | 361 | 1.759 | 6.89 | 0 |", ""),
        TABLE + "| GPU | long | 1963 | 8.470 | 7.71 | 0 |\n",
        TABLE.replace(
            "| Single ANE + GPU | long | 1963", "| Single ANE + GPU | long | 2048"
        ),
        TABLE.replace("8.189", "nan"),
        TABLE.replace("8.189", "0"),
        TABLE.replace("6.89", "-1"),
    ],
)
def test_rejects_incomplete_or_invalid_comparisons(tmp_path, text):
    source = tmp_path / "results.md"
    source.write_text(text)
    with pytest.raises(ValueError):
        read_rows(source)
