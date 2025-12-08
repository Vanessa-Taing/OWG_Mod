import os
import json
from datetime import datetime
from typing import List, Dict, Optional, Any
from collections import defaultdict
import hashlib

MODEL_CONFIGS = {
    "gpt-4o": {
        "model_name": "gpt-4o",
        "category": "large_vlm",
        "expected_cost_per_call": 0.006
    },
    "gpt-4o-mini": {
        "model_name": "gpt-4o-mini", 
        "category": "small_vlm",
        "expected_cost_per_call": 0.0015
    }
}

def generate_experiment_id(seed, config_path, timestamp):
    """Generate unique experiment ID from seed, config, and timestamp"""
    config_hash = hashlib.md5(config_path.encode()).hexdigest()[:8]
    ts_short = timestamp.split('T')[0].replace('-', '')  # YYYYMMDD
    return f"{ts_short}_{config_hash}_{seed}"

# ADD this helper function before ExperimentTracker class:
def detect_experiment_group(prompt_names):
    """Detect if experiment uses baseline or uncertainty-aware prompts"""
    # prompt_names is dict like {"grounder": "...", "planner": "...", "ranker": "..."}
    prompt_values = list(prompt_names.values())
    
    # Check if ANY prompt has '_base' suffix
    if any("_base" in p for p in prompt_values):
        return "baseline"
    else:
        return "uncertainty_aware"
    
# ADD this helper function:
def estimate_scenario_difficulty(planner_metadata, n_objects):
    """Simple proxy for scenario difficulty based on planning complexity"""
    difficulty_score = 0
    
    # Factor 1: Number of objects (normalized to 0-1)
    difficulty_score += min(n_objects / 20.0, 1.0) * 0.3
    
    # Factor 2: Check if plan has 'remove' actions (occlusion present)
    # This is extracted from planner's reasoning if available
    if planner_metadata and len(planner_metadata) > 0:
        latest_plan = planner_metadata[-1]
        
        # Check uncertainty description for occlusion keywords
        uncertainty_desc = latest_plan.get("uncertainty_description", "")
        if any(word in uncertainty_desc.lower() for word in ["cover", "block", "occlu", "obstruct"]):
            difficulty_score += 0.4
        
        # Check confidence - low confidence = harder scenario
        conf = latest_plan.get("confidence", 1.0)
        if conf != -1 and conf < 0.7:
            difficulty_score += 0.3
    
    # Map to categorical labels
    if difficulty_score < 0.3:
        return "easy", difficulty_score
    elif difficulty_score < 0.6:
        return "medium", difficulty_score
    else:
        return "hard", difficulty_score
    
class ExperimentTracker:
    def __init__(self, experiment_id=None, experiment_group="uncertainty_aware"): # ADD PARAMS
        self.experiment_id = experiment_id  # ADD
        self.experiment_group = experiment_group  # ADD
        self.metadata = defaultdict(list)  # module_name: [metadata_step1, metadata_step2, ...]
        self.model_settings = {}           # e.g., from get_model_params()
        self.model_category = []          # e.g., "large_vlm", "small_vlm"
        self.prompt_name = {}         # prompt name
        self.n_objects = 0               # number of objects in the scenario
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
        self.model_category = self._compute_model_categories()

    def _compute_model_categories(self):
        """Extract unique model categories from all stages"""
        categories = []
        for stage_name, stage_config in self.model_settings.items():
            model_name = stage_config.get("model_name", "unknown")
            category = MODEL_CONFIGS.get(model_name, {}).get("category", "unknown")
            if category not in categories and category != "unknown":
                categories.append(category)
        
        # Return sorted list for consistency
        return sorted(categories) if categories else ["unknown"]
    
    def set_prompt_variants(self, variants: List[str]):
        self.prompt_variants = variants

    def set_prompt_name(self, name: str):
        self.prompt_name = name

    def set_n_objects(self, n: int):
        self.n_objects = n

    def get_summary(self):
        return {
            "metadata": dict(self.metadata),
            "model": self.model_settings,
            "model_category": self.model_category,
            # "prompt_variants": self.prompt_variants
            "prompt_name": self.prompt_name,
            "n_objects": self.n_objects
        }

    def get_prompt_variants(self):
        return str(self.prompt_variants)    
    
    def get_model_category(self):
        """Returns the computed model category list"""
        return self.model_category

    def save_uncertainty_log(self, tracker_summary, save_path="logs/uncertainty_logs.jsonl"):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        record = {
            "experiment_id": self.experiment_id,  # ADD
            "experiment_group": self.experiment_group,  # ADD
            "timestamp": datetime.now().isoformat(),
            "metadata": tracker_summary.get("metadata", {}),
            "model": tracker_summary.get("model", {}),
            # "prompt_variants": tracker_summary.get("prompt_variants", {}),
            "model_category": tracker_summary.get("model_category", []),  # ADD THIS
            "prompt_name": tracker_summary.get("prompt_name", {}),
            "n_objects": tracker_summary.get("n_objects", {})
        }
        with open(save_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def extract_uncertainty_snapshot(self): #ADD THIS METHOD
        """Extract latest uncertainty values for linking to grasp outcome"""
        snapshot = {}
        
        # Get most recent metadata from each module
        for module_name in ["grounder", "planner", "ranker"]:
            if module_name in self.metadata and self.metadata[module_name]:
                latest = self.metadata[module_name][-1]  # Most recent
                
                # Extract confidence if present
                if "confidence" in latest:
                    conf = latest["confidence"]
                    snapshot[f"{module_name}_confidence"] = conf if conf != -1 else None
                
                # Extract entropy if present
                if "entropy" in latest:
                    snapshot[f"{module_name}_entropy"] = latest["entropy"]
        
        return snapshot if snapshot else None

class GraspStatsTracker(ExperimentTracker):
    def __init__(self, log_path: str = "logs/experiment_metrics.jsonl", experiment_id=None, experiment_group="uncertainty_aware"):  # ADD PARAMS):
        super().__init__(experiment_id, experiment_group)  # MODIFY to pass params
        self.grasp_log = []
        self.total_grasps = 0
        self.retries = 0
        self.successful_grasps = 0
        self.per_object_stats = defaultdict(lambda: {"success": 0, "total": 0})
        self.log_path = log_path
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def record_grasp(self, success: bool, object_id: int, position: List[float],
                     grasp_type: str = "2D",
                     grasp_index: Optional[int] = None, retries: int = 0,
                     additional_info: Optional[Dict[str, Any]] = None, 
                     uncertainty_snapshot: Optional[Dict[str, float]] = None):  # ADD THIS PARAM
        """Record a single grasp event and append it to both memory and file."""
        self.total_grasps += 1
        self.retries += retries
        if success:
            self.successful_grasps += 1
            self.per_object_stats[object_id]["success"] += 1
        self.per_object_stats[object_id]["total"] += 1

        entry = {
            "experiment_id": self.experiment_id,  # ADD
            "experiment_group": self.experiment_group,  # ADD
            "timestamp": datetime.now().isoformat(), # "timestamp": datetime.now().isoformat(timespec="seconds"),
            "object_id": object_id,
            "position": position,
            "success": success,
            "grasp_type": grasp_type,
            "grasp_index": grasp_index,
            "retries": retries,
        }

        if uncertainty_snapshot:
            entry["uncertainty_at_decision"] = uncertainty_snapshot

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
