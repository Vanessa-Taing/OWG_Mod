import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on sys.path so that `owg_mod` can be imported
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from owg_mod import app_utils


# ------------------------
# Unit tests
# ------------------------


def test_safe_column_exists_true_and_false():
    df = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [None, None, None],
        }
    )

    assert app_utils.safe_column_exists(df, "a") is True
    # Column exists but all NaN -> should be False
    assert app_utils.safe_column_exists(df, "b") is False
    # Column does not exist
    assert app_utils.safe_column_exists(df, "c") is False


def test_clean_confidence_data_replaces_minus_one_with_nan():
    df = pd.DataFrame(
        {
            "grounder_confidence": [0.1, -1, 0.9],
            "planner_confidence": [-1, 0.5, -1],
            "other": [1, 2, 3],
        }
    )

    cleaned = app_utils.clean_confidence_data(df)

    assert np.isnan(cleaned.loc[1, "grounder_confidence"])
    assert np.isnan(cleaned.loc[0, "planner_confidence"])
    assert np.isnan(cleaned.loc[2, "planner_confidence"])
    # Non-confidence column untouched
    pd.testing.assert_series_equal(cleaned["other"], df["other"])


def test_calculate_overall_confidence_uses_mean_across_stages():
    df = pd.DataFrame(
        {
            "grounder_confidence": [0.2, -1],
            "planner_confidence": [0.8, 0.6],
        }
    )

    result = app_utils.calculate_overall_confidence(df)

    # First row: (0.2 + 0.8) / 2
    assert pytest.approx(result.loc[0, "overall_confidence"], rel=1e-6) == 0.5
    # Second row: grounder_confidence == -1 -> converted to NaN, so mean is just 0.6
    assert pytest.approx(result.loc[1, "overall_confidence"], rel=1e-6) == 0.6


def test_filter_dataframe_basic_filters():
    df = pd.DataFrame(
        {
            "batch_id": ["b1", "b2", None],
            "experiment_group": ["g1", "g2", None],
            "prompt_type": ["p1", "p2", None],
            "grounder_model": ["m1", "m2", None],
            "success": [True, False, None],
        }
    )

    filters = {
        "batch_ids": ["b1"],
        "experiment_groups": ["g1"],
        "prompt_types": ["p1"],
        "model_names": ["m1"],
        "success_filter": "Success",
    }

    filtered = app_utils.filter_dataframe(df, filters)
    # Only the first row should remain
    assert len(filtered) == 1
    row = filtered.iloc[0]
    assert row["batch_id"] == "b1"
    assert row["experiment_group"] == "g1"
    assert row["prompt_type"] == "p1"
    assert row["grounder_model"] == "m1"
    assert row["success"] is True


def test_filter_dataframe_model_category_and_query_and_ranges():
    df = pd.DataFrame(
        {
            "model_category": ["vision,llm", "policy", None],
            "date": pd.to_datetime(["2025-01-01", "2025-01-02", None]).date,
            "n_objects": [3, 7, None],
            "query": ["pick red can", "grasp hammer", "other"],
        }
    )

    filters = {
        "model_categories": ["vision"],
        "date_range": (pd.to_datetime("2025-01-01").date(), pd.to_datetime("2025-01-01").date()),
        "n_objects_range": (1, 5),
        "query_search": "red",
    }

    filtered = app_utils.filter_dataframe(df, filters)
    assert len(filtered) == 1
    assert filtered.iloc[0]["model_category"] == "vision,llm"


def test_calculate_success_rate_overall_and_grouped():
    df = pd.DataFrame(
        {
            "group": ["a", "a", "b", "b"],
            "success": [True, False, True, True],
        }
    )

    overall = app_utils.calculate_success_rate(df)
    assert overall.iloc[0]["successes"] == 3  # True counts
    assert overall.iloc[0]["total"] == 4
    assert pytest.approx(overall.iloc[0]["success_rate"]) == 0.75

    grouped = app_utils.calculate_success_rate(df, group_by="group")
    assert set(grouped["group"]) == {"a", "b"}
    a_row = grouped[grouped["group"] == "a"].iloc[0]
    b_row = grouped[grouped["group"] == "b"].iloc[0]
    assert a_row["successes"] == 1
    assert a_row["total"] == 2
    assert pytest.approx(a_row["success_rate"]) == 0.5
    assert b_row["successes"] == 2
    assert b_row["total"] == 2
    assert pytest.approx(b_row["success_rate"]) == 1.0


def test_perform_ttest_and_nonparametric():
    # Normal-like data should use t-test; we just check that it returns finite values
    g1 = pd.Series(np.random.normal(loc=0.0, scale=1.0, size=50))
    g2 = pd.Series(np.random.normal(loc=1.0, scale=1.0, size=50))

    stat, p_value, test_type = app_utils.perform_ttest(g1, g2)
    assert np.isfinite(stat)
    assert 0.0 <= p_value <= 1.0
    assert test_type in {"Independent t-test", "Mann-Whitney U"}


def test_perform_anova_and_posthoc():
    g1 = pd.Series([1, 2, 3, 4])
    g2 = pd.Series([2, 3, 4, 5])
    g3 = pd.Series([10, 11, 12, 13])

    f_stat, p_value, posthoc = app_utils.perform_anova(
        [g1, g2, g3], ["g1", "g2", "g3"]
    )

    assert np.isfinite(f_stat)
    assert 0.0 <= p_value <= 1.0
    # Posthoc should contain pairwise rows
    assert set(posthoc.columns) == {
        "group1",
        "group2",
        "statistic",
        "p_value",
        "significant",
    }
    assert len(posthoc) == 3  # 3 pairwise comparisons for 3 groups


def test_calculate_calibration_metrics_basic():
    df = pd.DataFrame(
        {
            "overall_confidence": [0.1, 0.9, 0.8, 0.2, 0.5, 0.6, 0.4, 0.3, 0.7, 0.95],
            "success": [0, 1, 1, 0, 0, 1, 0, 0, 1, 1],
        }
    )

    metrics = app_utils.calculate_calibration_metrics(df)

    assert "ece" in metrics
    assert "mce" in metrics
    assert "brier_score" in metrics
    assert "bin_data" in metrics
    assert isinstance(metrics["bin_data"], list)
    assert metrics["ece"] >= 0
    assert metrics["mce"] >= 0
    assert 0 <= metrics["brier_score"] <= 1


@pytest.mark.parametrize(
    "p, expected",
    [
        (np.nan, ""),
        (0.2, "ns"),
        (0.04, "*"),
        (0.009, "**"),
        (0.0009, "***"),
    ],
)
def test_get_significance_marker(p, expected):
    assert app_utils.get_significance_marker(p) == expected


def test_calculate_correlation_basic_and_insufficient_data():
    df = pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5],
            "y": [2, 4, 6, 8, 10],  # perfectly correlated
        }
    )

    corr, p_value = app_utils.calculate_correlation(df, "x", "y")
    assert pytest.approx(corr, rel=1e-6) == 1.0
    assert 0.0 <= p_value <= 1.0

    # Insufficient data case
    df_small = pd.DataFrame({"x": [1, 2], "y": [1, 2]})
    corr_small, p_small = app_utils.calculate_correlation(df_small, "x", "y")
    assert np.isnan(corr_small)
    assert np.isnan(p_small)


# ------------------------
# Integration tests
# ------------------------


def _write_jsonl(path: Path, records):
    with path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def test_load_experiment_metrics_and_load_uncertainty_logs_and_merge_logs(tmp_path, monkeypatch):
    # Prepare experiment_metrics.jsonl
    metrics_records = [
        {
            "experiment_id": "exp1",
            "timestamp": "2025-01-01T00:00:00",
            "success": True,
            "retries": 1,
            "grasp_type": "power",
            "scenario_difficulty": "easy",
            "difficulty_score": 0.2,
        },
        {
            "experiment_id": "exp1",
            "timestamp": "2025-01-01T00:01:00",
            "success": False,
            "retries": 2,
            "grasp_type": "precision",
            "scenario_difficulty": "easy",
            "difficulty_score": 0.2,
        },
    ]

    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    metrics_path = logs_dir / "experiment_metrics.jsonl"
    _write_jsonl(metrics_path, metrics_records)

    # Prepare uncertainty_logs.jsonl
    uncertainty_records = [
        {
            "experiment_id": "exp1",
            "experiment_group": "g1",
            "timestamp": "2025-01-01T00:00:30",
            "n_objects": 5,
            "model_category": ["vision", "llm"],
            "model": {
                "grounder": {"model_name": "gpt-4o"},
                "planner": {"model_name": "gpt-4o"},
            },
            "prompt_name": {
                "grounder": "grounder_prompt_v1",
                "planner": "planner_prompt_v1",
            },
            "metadata": {
                "grounder": [
                    {"confidence": 0.7, "entropy": 0.2},
                    {"confidence": 0.9, "entropy": 0.1},
                ],
                "planner": [
                    {"confidence": 0.6, "entropy": 0.3},
                ],
            },
        }
    ]

    uncertainty_path = logs_dir / "uncertainty_logs.jsonl"
    _write_jsonl(uncertainty_path, uncertainty_records)

    # Monkeypatch expanduser so that "~/OWG" points to our tmp_path / "OWG"
    owg_dir = tmp_path / "OWG"
    owg_logs_dir = owg_dir / "logs"
    owg_logs_dir.mkdir(parents=True)

    # Move the prepared files into that structure
    os.replace(metrics_path, owg_logs_dir / "experiment_metrics.jsonl")
    os.replace(uncertainty_path, owg_logs_dir / "uncertainty_logs.jsonl")

    original_expanduser = os.path.expanduser

    def _fake_expanduser(path: str) -> str:
        if path.startswith("~"):
            return str(tmp_path / path.lstrip("~/"))
        return original_expanduser(path)

    monkeypatch.setattr(os.path, "expanduser", _fake_expanduser)

    metrics_df = app_utils.load_experiment_metrics()
    uncertainty_df = app_utils.load_uncertainty_logs()

    assert not metrics_df.empty
    assert not uncertainty_df.empty
    assert "timestamp" in metrics_df.columns
    assert "timestamp" in uncertainty_df.columns
    assert "n_objects" in uncertainty_df.columns

    # Prepare a simple batch_df compatible with merge_logs
    batch_df = pd.DataFrame(
        {
            "experiment_id": ["exp1"],
            "batch_id": ["batch_1"],
            "query": ["pick object"],
            "prompt_type": ["baseline"],
            "n_objects": [5],
            "model_category": ["vision,llm"],
            "attempts": [3],
        }
    )

    merged = app_utils.merge_logs(uncertainty_df, batch_df, metrics_df)
    assert not merged.empty
    assert "batch_id" in merged.columns
    assert "query" in merged.columns
    assert "attempts" in merged.columns
    assert "success" in merged.columns
    assert "grasp_retries" in merged.columns
    # From metrics aggregation: last success is False (check logical value, not identity)
    assert bool(merged["success"].iloc[0]) is False
    assert merged["grasp_retries"].iloc[0] == 3


def test_load_batch_logs_from_experiments_dir(tmp_path, monkeypatch):
    # Create ~/OWG/experiments/batch_exp_XXXX/batch_results.jsonl under tmp_path
    owg_dir = tmp_path / "OWG"
    experiments_dir = owg_dir / "experiments"
    batch_dir = experiments_dir / "batch_exp_20250101_000000"
    batch_dir.mkdir(parents=True)

    records = [
        {
            "experiment_id": "exp1",
            "timestamp": "2025-01-01T00:00:00",
            "query": "pick object",
            "prompt_type": "baseline",
        },
        {
            "experiment_id": "exp2",
            "timestamp": "2025-01-01T00:01:00",
            "query": "pick another object",
            "prompt_type": "confidence",
        },
    ]

    _write_jsonl(batch_dir / "batch_results.jsonl", records)

    original_expanduser = os.path.expanduser

    def _fake_expanduser(path: str) -> str:
        if path.startswith("~"):
            return str(tmp_path / path.lstrip("~/"))
        return original_expanduser(path)

    monkeypatch.setattr(os.path, "expanduser", _fake_expanduser)

    batch_df = app_utils.load_batch_logs()
    assert not batch_df.empty
    assert set(batch_df["experiment_id"]) == {"exp1", "exp2"}
    assert "batch_id" in batch_df.columns
    # batch_id should be the directory name
    assert batch_df["batch_id"].nunique() == 1
    assert batch_df["batch_id"].iloc[0] == batch_dir.name


# ------------------------
# System-style end-to-end test
# ------------------------


def test_end_to_end_pipeline_on_sample_logs(tmp_path, monkeypatch):
    """
    High-level smoke test that exercises the typical workflow:
    - load_experiment_metrics
    - load_uncertainty_logs
    - load_batch_logs
    - merge_logs
    - calculate_overall_confidence
    - calculate_success_rate
    """
    # Prepare directory structure under a fake "~/OWG" mapped to tmp_path / "OWG"
    owg_dir = tmp_path / "OWG"
    logs_dir = owg_dir / "logs"
    experiments_dir = owg_dir / "experiments"
    logs_dir.mkdir(parents=True)
    experiments_dir.mkdir(parents=True)

    # Metrics
    metrics_records = [
        {
            "experiment_id": "exp1",
            "timestamp": "2025-01-01T00:00:00",
            "success": True,
            "retries": 1,
        },
        {
            "experiment_id": "exp2",
            "timestamp": "2025-01-01T00:01:00",
            "success": False,
            "retries": 0,
        },
    ]
    _write_jsonl(logs_dir / "experiment_metrics.jsonl", metrics_records)

    # Uncertainty logs
    uncertainty_records = [
        {
            "experiment_id": "exp1",
            "experiment_group": "g1",
            "timestamp": "2025-01-01T00:00:30",
            "n_objects": 5,
            "model_category": ["vision"],
            "metadata": {
                "grounder": [{"confidence": 0.8, "entropy": 0.2}],
                "planner": [{"confidence": 0.7, "entropy": 0.3}],
            },
        },
        {
            "experiment_id": "exp2",
            "experiment_group": "g2",
            "timestamp": "2025-01-01T00:02:00",
            "n_objects": 3,
            "model_category": ["vision"],
            "metadata": {
                "grounder": [{"confidence": 0.3, "entropy": 0.6}],
                "planner": [{"confidence": 0.4, "entropy": 0.5}],
            },
        },
    ]
    _write_jsonl(logs_dir / "uncertainty_logs.jsonl", uncertainty_records)

    # Batch logs
    batch_dir = experiments_dir / "batch_exp_20250101_000000"
    batch_dir.mkdir()
    batch_records = [
        {
            "experiment_id": "exp1",
            "timestamp": "2025-01-01T00:00:00",
            "query": "pick object",
            "prompt_type": "baseline",
            "n_objects": 5,
            "model_category": "vision",
            "attempts": 2,
        },
        {
            "experiment_id": "exp2",
            "timestamp": "2025-01-01T00:01:00",
            "query": "pick another",
            "prompt_type": "confidence",
            "n_objects": 3,
            "model_category": "vision",
            "attempts": 1,
        },
    ]
    _write_jsonl(batch_dir / "batch_results.jsonl", batch_records)

    # Monkeypatch expanduser so that "~/OWG" resolves inside tmp_path
    original_expanduser = os.path.expanduser

    def _fake_expanduser(path: str) -> str:
        if path.startswith("~"):
            return str(tmp_path / path.lstrip("~/"))
        return original_expanduser(path)

    monkeypatch.setattr(os.path, "expanduser", _fake_expanduser)

    # Run the full flow
    metrics_df = app_utils.load_experiment_metrics()
    uncertainty_df = app_utils.load_uncertainty_logs()
    batch_df = app_utils.load_batch_logs()

    merged = app_utils.merge_logs(uncertainty_df, batch_df, metrics_df)
    assert not merged.empty
    merged = app_utils.calculate_overall_confidence(merged)

    # We should have overall_confidence and success columns
    assert "overall_confidence" in merged.columns
    assert "success" in merged.columns

    # Calculate success rate by experiment_group as an example aggregation
    success_by_group = app_utils.calculate_success_rate(
        merged, group_by="experiment_group"
    )
    assert not success_by_group.empty
    assert set(success_by_group["experiment_group"]) == {"g1", "g2"}


