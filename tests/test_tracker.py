import json
from pathlib import Path

import numpy as np
import pytest

from owg_mod import tracker


# ------------------------
# Helper function tests
# ------------------------


def test_generate_experiment_id_is_deterministic_and_formatted():
    seed = 42
    config_path = "config/exp_config.yaml"
    timestamp = "2025-01-02T12:34:56"

    exp_id_1 = tracker.generate_experiment_id(seed, config_path, timestamp)
    exp_id_2 = tracker.generate_experiment_id(seed, config_path, timestamp)

    # Deterministic for same inputs
    assert exp_id_1 == exp_id_2

    # Basic format checks: YYYYMMDD_<hash>_<seed>
    parts = exp_id_1.split("_")
    assert len(parts) == 3
    date_part, hash_part, seed_part = parts
    assert date_part == "20250102"
    assert len(hash_part) == 8
    assert seed_part == str(seed)


def test_detect_experiment_group_baseline_vs_uncertainty():
    baseline_prompts = {
        "grounder": "grounder_base_v1",
        "planner": "planner_base_v1",
        "ranker": "ranker_base_v1",
    }
    uncertainty_prompts = {
        "grounder": "grounder_unc_v1",
        "planner": "planner_unc_v1",
        "ranker": "ranker_unc_v1",
    }

    assert tracker.detect_experiment_group(baseline_prompts) == "baseline"
    assert tracker.detect_experiment_group(uncertainty_prompts) == "uncertainty_aware"


@pytest.mark.parametrize(
    "n_objects, conf, occluded, expected_label",
    [
        (1, 1.0, False, "easy"),
        # Higher object count with good confidence but no occlusion -> medium
        (20, 0.8, False, "medium"),
        # Many objects, low confidence, occlusion -> hard
        (15, 0.5, True, "hard"),
    ],
)
def test_estimate_scenario_difficulty(n_objects, conf, occluded, expected_label):
    meta = [
        {
            "confidence": conf,
            "uncertainty_description": "object is clearly visible",
        }
    ]
    if occluded:
        meta[0]["uncertainty_description"] = "target is partially occluded and blocked"

    label, score = tracker.estimate_scenario_difficulty(meta, n_objects)
    assert label == expected_label
    assert 0.0 <= score <= 1.0


# ------------------------
# ExperimentTracker tests
# ------------------------


def test_experiment_tracker_model_category_and_summary():
    et = tracker.ExperimentTracker(experiment_id="exp1", experiment_group="baseline")

    # Set model settings with known MODEL_CONFIGS
    et.set_model_settings(
        {
            "grounder": {"model_name": "gpt-4o"},
            "planner": {"model_name": "gpt-4o-mini"},
        }
    )
    categories = et.get_model_category()
    # Should map to known categories from MODEL_CONFIGS
    assert "large_vlm" in categories
    assert "small_vlm" in categories

    # Set prompts and n_objects
    et.set_prompt_name(
        {
            "grounder": "grounder_base_v1",
            "planner": "planner_base_v1",
        }
    )
    et.set_n_objects(5)

    summary = et.get_summary()
    assert summary["model_category"] == categories
    assert summary["prompt_name"]["grounder"] == "grounder_base_v1"
    assert summary["n_objects"] == 5


def test_experiment_tracker_metadata_and_uncertainty_snapshot():
    et = tracker.ExperimentTracker(experiment_id="exp2", experiment_group="uncertainty_aware")

    # Record multiple steps for grounder and planner
    et.set_metadata({"confidence": 0.3, "entropy": 0.9}, module_name="grounder")
    et.set_metadata({"confidence": 0.7, "entropy": 0.2}, module_name="grounder")
    et.set_metadata({"confidence": 0.5, "entropy": 0.5}, module_name="planner")

    snapshot = et.extract_uncertainty_snapshot()
    # Should reflect the most recent entries per module
    assert snapshot["grounder_confidence"] == 0.7
    assert snapshot["grounder_entropy"] == 0.2
    assert snapshot["planner_confidence"] == 0.5
    assert snapshot["planner_entropy"] == 0.5


def test_experiment_tracker_save_uncertainty_log(tmp_path):
    et = tracker.ExperimentTracker(experiment_id="exp3", experiment_group="baseline")
    et.set_model_settings({"grounder": {"model_name": "gpt-4o"}})
    et.set_prompt_name({"grounder": "grounder_base_v1"})
    et.set_n_objects(3)

    summary = et.get_summary()
    log_path = tmp_path / "logs" / "uncertainty_logs.jsonl"

    et.save_uncertainty_log(summary, save_path=str(log_path))

    assert log_path.is_file()
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])

    assert record["experiment_id"] == "exp3"
    assert record["experiment_group"] == "baseline"
    assert record["model"]["grounder"]["model_name"] == "gpt-4o"
    assert record["prompt_name"]["grounder"] == "grounder_base_v1"
    assert record["n_objects"] == 3


# ------------------------
# GraspStatsTracker tests
# ------------------------


def test_grasp_stats_tracker_record_and_rates(tmp_path):
    log_path = tmp_path / "logs" / "experiment_metrics.jsonl"
    gst = tracker.GraspStatsTracker(
        log_path=str(log_path), experiment_id="exp4", experiment_group="uncertainty_aware"
    )

    # Simulate two grasps on different objects
    gst.record_grasp(
        success=True,
        object_id=1,
        position=[0.0, 0.1, 0.2],
        grasp_type="3D",
        retries=1,
        additional_info={"info": "first grasp"},
        uncertainty_snapshot={"grounder_confidence": 0.8},
    )
    gst.record_grasp(
        success=False,
        object_id=2,
        position=[0.3, 0.4, 0.5],
        grasp_type="2D",
        retries=0,
    )

    # In-memory stats
    assert gst.total_grasps == 2
    assert gst.successful_grasps == 1
    assert gst.retries == 1
    assert pytest.approx(gst.get_success_rate()) == 0.5

    per_obj = gst.get_success_rate_per_object()
    assert per_obj[1] == 1.0
    assert per_obj[2] == 0.0

    # File written
    assert log_path.is_file()
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    rec0 = json.loads(lines[0])
    rec1 = json.loads(lines[1])
    assert rec0["experiment_id"] == "exp4"
    assert rec0["experiment_group"] == "uncertainty_aware"
    assert rec0["success"] is True
    assert rec0["uncertainty_at_decision"]["grounder_confidence"] == 0.8
    assert rec1["success"] is False


def test_grasp_stats_tracker_reset(tmp_path):
    log_path = tmp_path / "logs" / "experiment_metrics.jsonl"
    gst = tracker.GraspStatsTracker(
        log_path=str(log_path), experiment_id="exp5", experiment_group="baseline"
    )

    gst.record_grasp(
        success=True,
        object_id=1,
        position=[0.0, 0.0, 0.0],
    )
    assert gst.total_grasps == 1
    assert gst.successful_grasps == 1

    gst.reset()

    # After reset, stats should be cleared
    assert gst.total_grasps == 0
    assert gst.successful_grasps == 0
    assert gst.retries == 0
    assert gst.get_log() == []

    # Log file path should still exist (directory created), but we don't enforce clearing file
    assert Path(log_path).parent.is_dir()


