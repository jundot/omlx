# SPDX-License-Identifier: Apache-2.0
"""Tests for persistent model catalog metadata."""

from pathlib import Path

from omlx.model_catalog import ModelCatalog


def test_record_download_preserves_existing_benchmark_summaries(tmp_path: Path):
    catalog = ModelCatalog(tmp_path)
    catalog.update_perf_summary(
        "gemma-4-31B-it",
        "perf-1",
        {"result_id": "perf-1", "tg_tps": 12.3},
    )
    catalog.replace_accuracy_summaries(
        {
            "gemma-4-31B-it": {
                "last_accuracy_result_id": "acc-1",
                "best_accuracy_summary": {"result_id": "acc-1", "accuracy": 0.9},
                "accuracy_summaries_by_benchmark": {
                    "humaneval": {"result_id": "acc-1", "accuracy": 0.9}
                },
            }
        }
    )

    model_path = tmp_path / "models" / "google" / "gemma-4-31B-it"
    catalog.record_download(
        model_id="gemma-4-31B-it",
        path=model_path,
        source="hf",
        repo_id="google/gemma-4-31B-it",
        remote_revision="abc123",
    )

    entry = catalog.get_public("gemma-4-31B-it")
    assert entry["source"] == "hf"
    assert entry["repo_id"] == "google/gemma-4-31B-it"
    assert entry["last_perf_result_id"] == "perf-1"
    assert entry["best_perf_summary"]["tg_tps"] == 12.3
    assert entry["last_accuracy_result_id"] == "acc-1"
    assert entry["best_accuracy_summary"]["accuracy"] == 0.9
    assert entry["accuracy_summaries_by_benchmark"]["humaneval"]["accuracy"] == 0.9


def test_reconcile_infers_hf_source_for_nested_huggingface_local_dir(tmp_path: Path):
    catalog = ModelCatalog(tmp_path)
    model_path = tmp_path / "models" / "google" / "gemma-4-31B-it"
    (model_path / ".cache" / "huggingface" / "download").mkdir(parents=True)

    catalog.reconcile(
        [
            {
                "id": "gemma-4-31B-it",
                "model_path": str(model_path),
            }
        ]
    )

    entry = catalog.get_public("gemma-4-31B-it")
    assert entry["source"] == "hf"
    assert entry["provider"] == "huggingface"
    assert entry["repo_id"] == "google/gemma-4-31B-it"
    assert entry["update_status"] == "not_checked"


def test_reconcile_upgrades_existing_local_entry_from_huggingface_cache(
    tmp_path: Path,
):
    catalog = ModelCatalog(tmp_path)
    old_path = tmp_path / "models" / "gemma-4-31B-it"
    catalog.reconcile(
        [
            {
                "id": "gemma-4-31B-it",
                "model_path": str(old_path),
            }
        ]
    )
    assert catalog.get_public("gemma-4-31B-it")["source"] == "local"

    nested_path = tmp_path / "models" / "google" / "gemma-4-31B-it"
    (nested_path / ".cache" / "huggingface" / "download").mkdir(parents=True)
    catalog.reconcile(
        [
            {
                "id": "gemma-4-31B-it",
                "model_path": str(nested_path),
            }
        ]
    )

    entry = catalog.get_public("gemma-4-31B-it")
    assert entry["source"] == "hf"
    assert entry["provider"] == "huggingface"
    assert entry["repo_id"] == "google/gemma-4-31B-it"
