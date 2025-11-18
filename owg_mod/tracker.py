import os
import json
from datetime import datetime
from typing import List, Dict, Optional, Any
from collections import defaultdict

class ExperimentTracker:
    def __init__(self):
        self.metadata = defaultdict(list)  # module_name: [metadata_step1, metadata_step2, ...]
        self.model_settings = {}           # e.g., from get_model_params()
        self.prompt_variants = []          # list of prompt types
        self.step_counters = defaultdict(int)  # internal counter per module for iteration tracking

    def set_metadata(self, metadata_dict: Dict[str, Any], module_name: Optional[str] = None):
        if module_name is None:
            self.metadata = metadata_dict
        else:
            step = self.step_counters[module_name]
            metadata_dict["step"] = step
            self.metadata[module_name].append(metadata_dict)
            self.step_counters[module_name] += 1

    def set_model_settings(self, settings_dict: Dict[str, Any]):
        self.model_settings = settings_dict

    def set_prompt_variants(self, variants: List[str]):
        self.prompt_variants = variants

    def get_summary(self):
        return {
            "metadata": dict(self.metadata),
            "model": self.model_settings,
            "prompt_variants": self.prompt_variants
        }
    
    def save_uncertainty_log(self, tracker_summary, save_path="logs/uncertainty_logs.jsonl"):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        record = {
            "timestamp": datetime.now().isoformat(),
            "metadata": tracker_summary.get("metadata", {}),
            "model": tracker_summary.get("model", {}),
            "prompt_variants": tracker_summary.get("prompt_variants", {}),
        }
        with open(save_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def get_prompt_variants(self):
        return str(self.prompt_variants)


class GraspStatsTracker(ExperimentTracker):
    def __init__(self, log_path: str = "logs/experiment_metrics.jsonl"):
        super().__init__()
        self.grasp_log = []
        self.total_grasps = 0
        self.retries = 0
        self.successful_grasps = 0
        self.per_object_stats = defaultdict(lambda: {"success": 0, "total": 0})
        self.log_path = log_path

        # Ensure the logs directory exists
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def record_grasp(self, success: bool, object_id: int, position: List[float],
                     grasp_type: str = "2D",
                     grasp_index: Optional[int] = None, retries: int = 0,
                     additional_info: Optional[Dict[str, Any]] = None):
        """Record a single grasp event and append it to both memory and file."""
        self.total_grasps += 1
        self.retries += retries
        if success:
            self.successful_grasps += 1
            self.per_object_stats[object_id]["success"] += 1
        self.per_object_stats[object_id]["total"] += 1

        entry = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "object_id": object_id,
            "position": position,
            "success": success,
            "grasp_type": grasp_type,
            "grasp_index": grasp_index,
            "retries": retries,
        }
        if additional_info:
            entry.update(additional_info)

        # Save to in-memory log
        self.grasp_log.append(entry)

        # Write to file (JSON Lines)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def get_success_rate(self) -> float:
        return self.successful_grasps / self.total_grasps if self.total_grasps else 0.0

    def get_success_rate_per_object(self) -> Dict[int, float]:
        return {
            obj_id: s["success"] / s["total"] if s["total"] else 0.0
            for obj_id, s in self.per_object_stats.items()
        }

    def get_log(self) -> List[Dict[str, Any]]:
        return self.grasp_log

    def reset(self):
        """Reset in-memory stats and optionally clear the file log."""
        self.__init__(self.log_path)
