#!/usr/bin/env python3
"""
Production Batch Experiment Runner for OWG Uncertainty-Aware Research
=====================================================================

Systematically runs experiments across:
- Multiple random seeds
- Different queries and object counts
- Baseline vs uncertainty-aware prompts
- GPT-4o vs GPT-4o-mini models

Features:
- Progress tracking and resume capability
- Error handling and retry logic
- Automatic result aggregation
- Experiment tracking with unique IDs

Usage:
    python notebooks/batch_experiments.py --mode quick
    python notebooks/batch_experiments.py --mode baseline_vs_uncertainty
    python notebooks/batch_experiments.py --mode model_comparison --models gpt-4o gpt-4o-mini
    python notebooks/batch_experiments.py --mode full
"""

import argparse
import subprocess
import json
import os
import yaml
import time
from datetime import datetime
from itertools import product
from pathlib import Path
import sys

# ============================================================================
# EXPERIMENT CONFIGURATIONS
# ============================================================================

SEEDS = [42, 123, 456]  # 5 different random seeds for reproducibility [789, 1024]

QUERIES = [
    "pick up the smallest object",
    "pick up the largest object", 
    "pick up the object with the most vibrant color",
    # "pick up the cylindrical object",
]

N_OBJECTS_RANGE = [5, 10, 15]  # Different clutter levels: low, medium, high

# Prompt configuration templates
PROMPT_CONFIGS = {
    "baseline": {
        "grounder": "referring_segmentation_base",
        "planner": "grasp_planning_base",
        "ranker": "grasp_ranking_base"
    },
    "confidence": {
        "grounder": "referring_segmentation_confidence",
        "planner": "grasp_planning_confidence", 
        "ranker": "grasp_ranking_confidence"
    },
    "hedging": {
        "grounder": "referring_segmentation_hedging",
        "planner": "grasp_planning_hedging",
        "ranker": "grasp_ranking_hedging"
    },
    "cautious": {
        "grounder": "referring_segmentation_cautious",
        "planner": "grasp_planning_cautious",
        "ranker": "grasp_ranking_cautious"
    },
    "uncertainty_description": {
        "grounder": "referring_segmentation_uncertainty_description",
        "planner": "grasp_planning_uncertainty_description",
        "ranker": "grasp_ranking_uncertainty_description"
    }
}

# Model configurations for RQ4 (SVLM comparison)
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

# ============================================================================
# CONFIG GENERATION
# ============================================================================

def generate_config_file(prompt_config, model_config, output_path):
    """
    Generate YAML config file from prompt and model configuration
    
    Args:
        prompt_config: Dict with keys 'grounder', 'planner', 'ranker'
        model_config: Dict with model_name and metadata
        output_path: Path to save config YAML
    
    Returns:
        Path to generated config file
    """
    config = {
        'image_size_h': 448,
        'image_size_w': 448,
        'image_crop': None,
        
        # Prompt directories
        'grounding_prompt_root_dir': './prompts/uncertainty_aware',
        'planning_prompt_root_dir': './prompts/uncertainty_aware',
        'grasping_prompt_root_dir': './prompts/uncertainty_aware',
        
        # Grounding configuration
        'grounding': {
            'prompt_name': prompt_config['grounder'],
            'prompt_template': 'Description: {user_input}',
            'request': {
                'model_name': model_config['model_name'],
                'temperature': 0.1,
                'n': 2,
                'max_tokens': 256,
                'logprobs': True,
                'detail': 'auto',
                'seed': 12
            },
            'include_raw_image': True,
            'use_subplot_prompt': False,
            'subplot_size': 224,
            'do_refine_marks': False,
            'refine_marks': {
                'maximum_hole_area': 0.01,
                'maximum_island_area': 0.01,
                'minimum_mask_area': 0.02,
                'maximum_mask_area': 1.0
            },
            'do_inctx': False,
            'inctx_prompt_name': None,
            'visualizer': {
                'label': {
                    'text_include': True,
                    'text_scale': 0.5,
                    'text_thickness': 2,
                    'text_padding': 2,
                    'text_position': 'TOP_CENTER'
                },
                'box': {
                    'box_include': False,
                    'box_thickness': 2
                },
                'mask': {
                    'mask_include': True,
                    'mask_opacity': 0.25
                },
                'polygon': {
                    'polygon_include': True,
                    'polygon_thickness': 2
                }
            }
        },
        
        # Planning configuration
        'planning': {
            'prompt_name': prompt_config['planner'],
            'prompt_template': 'Task instruction: "Target object {user_input}".',
            'request': {
                'model_name': model_config['model_name'],
                'temperature': 0.0,
                'n': 2,
                'max_tokens': 256,
                'logprobs': True,
                'detail': 'auto'
            },
            'response_format': 'json',
            'include_raw_image': False,
            'use_subplot_prompt': False,
            'subplot_size': 448,
            'do_refine_marks': False,
            'refine_marks': {
                'maximum_hole_area': 0.01,
                'maximum_island_area': 0.01,
                'minimum_mask_area': 0.02,
                'maximum_mask_area': 1.0
            },
            'do_inctx': False,
            'inctx_prompt_name': 'pyb/inctx_grasp_planning.pt',
            'visualizer': {
                'label': {
                    'text_include': True,
                    'text_scale': 0.5,
                    'text_thickness': 2,
                    'text_padding': 2,
                    'text_position': 'CENTER_OF_MASS'
                },
                'box': {
                    'box_include': False,
                    'box_thickness': 2
                },
                'mask': {
                    'mask_include': True,
                    'mask_opacity': 0.25
                },
                'polygon': {
                    'polygon_include': True,
                    'polygon_thickness': 1
                }
            }
        },
        
        # Grasping/Ranking configuration
        'grasping': {
            'prompt_name': prompt_config['ranker'],
            'prompt_template': 'Rank the grasp poses.',
            'request': {
                'model_name': model_config['model_name'],
                'temperature': 0.0,
                'n': 2,
                'max_tokens': 256,
                'logprobs': True,
                'detail': 'auto'
            },
            'crop_square_size': 196,
            'use_3d_prompt': False,
            'gripper_mesh_path': 'owg_robot/assets/robotiq_2f_140/robotiq_arg2f_140.obj',
            'use_subplot_prompt': True,
            'subplot_size': 224,
            'do_inctx': False,
            'inctx_prompt_name': 'pyb/inctx_grasp_ranking.pt',
            'visualizer': {
                'as_line': True,
                'line_thickness': 8,
                'grasp_colors': 'RED,GREEN',
                'with_gray': False,
                'label': {
                    'label_include': False,
                    'text_color': 'WHITE',
                    'text_rect_color': 'BLACK',
                    'text_padding': 2,
                    'text_thickness': 1,
                    'text_scale': 0.7,
                    'text_position': 'CENTER'
                },
                'box': {
                    'box_include': False,
                    'box_color': 'RED',
                    'box_thickness': 2
                },
                'mask': {
                    'mask_include': False,
                    'mask_color': 'RED',
                    'mask_opacity': 0.15
                },
                'polygon': {
                    'polygon_include': True,
                    'polygon_color': 'RED',
                    'polygon_thickness': 2
                }
            }
        }
    }
    
    # Write to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    return output_path

# ============================================================================
# EXPERIMENT EXECUTION
# ============================================================================

def run_single_experiment(seed, query, n_objects, config_path, output_dir, timeout=300, max_retries=2):
    """
    Run a single OWG experiment with retry logic
    
    Args:
        seed: Random seed
        query: User query string
        n_objects: Number of objects in scene
        config_path: Path to config YAML
        output_dir: Directory for outputs
        timeout: Timeout in seconds
        max_retries: Number of retry attempts on failure
    
    Returns:
        Dict with success status and metadata
    """
    cmd = [
        "python", "notebooks/owg_evaluation_pipeline.py",
        "--seed", str(seed),
        "--config", config_path,
        "--query", query,
        "--n-objects", str(n_objects),
        "--output-dir", output_dir,
        "--headless"
    ]
    
    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                print(f"   ⟳ Retry attempt {attempt}/{max_retries}...")
                time.sleep(2)  # Brief pause before retry
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=timeout,
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Run from OWG root
            )
            
            if result.returncode == 0:
                return {
                    "success": True,
                    "attempts": attempt + 1,
                    "stdout": result.stdout[-500:],  # Last 500 chars
                    "stderr": ""
                }
            else:
                if attempt == max_retries:
                    return {
                        "success": False,
                        "attempts": attempt + 1,
                        "stdout": result.stdout[-500:],
                        "stderr": result.stderr[-500:]
                    }
                # Otherwise continue to retry
                
        except subprocess.TimeoutExpired:
            if attempt == max_retries:
                return {
                    "success": False,
                    "attempts": attempt + 1,
                    "error": f"timeout after {timeout}s"
                }
        except Exception as e:
            if attempt == max_retries:
                return {
                    "success": False,
                    "attempts": attempt + 1,
                    "error": str(e)
                }
    
    return {"success": False, "error": "max retries exceeded"}

# ============================================================================
# PROGRESS TRACKING
# ============================================================================

def load_completed_experiments(batch_dir):
    """Load list of already-completed experiment IDs for resume capability"""
    results_file = os.path.join(batch_dir, "batch_results.jsonl")
    if not os.path.exists(results_file):
        return set()
    
    completed = set()
    with open(results_file, "r") as f:
        for line in f:
            try:
                record = json.loads(line)
                if record.get("success"):
                    exp_id = (
                        record["seed"],
                        record["query"], 
                        record["n_objects"],
                        record["prompt_type"],
                        record["model_name"]
                    )
                    completed.add(exp_id)
            except:
                pass
    
    return completed

def save_experiment_result(result_record, batch_dir):
    """Append experiment result to JSONL file"""
    results_file = os.path.join(batch_dir, "batch_results.jsonl")
    with open(results_file, "a") as f:
        f.write(json.dumps(result_record) + "\n")

# ============================================================================
# RESULT AGGREGATION
# ============================================================================

def generate_summary_report(batch_dir):
    """Generate summary statistics from batch results"""
    results_file = os.path.join(batch_dir, "batch_results.jsonl")
    
    if not os.path.exists(results_file):
        return None
    
    results = []
    with open(results_file, "r") as f:
        for line in f:
            try:
                results.append(json.loads(line))
            except:
                pass
    
    if not results:
        return None
    
    summary = {
        "total_experiments": len(results),
        "successful": sum(1 for r in results if r["success"]),
        "failed": sum(1 for r in results if not r["success"]),
        "by_prompt_type": {},
        "by_model": {},
        "by_n_objects": {},
        "average_runtime": None  # Could add timing if we track it
    }
    
    # Group by prompt type
    for prompt_type in set(r["prompt_type"] for r in results):
        subset = [r for r in results if r["prompt_type"] == prompt_type]
        summary["by_prompt_type"][prompt_type] = {
            "total": len(subset),
            "successful": sum(1 for r in subset if r["success"]),
            "success_rate": sum(1 for r in subset if r["success"]) / len(subset) if subset else 0
        }
    
    # Group by model
    for model_name in set(r["model_name"] for r in results):
        subset = [r for r in results if r["model_name"] == model_name]
        summary["by_model"][model_name] = {
            "total": len(subset),
            "successful": sum(1 for r in subset if r["success"]),
            "success_rate": sum(1 for r in subset if r["success"]) / len(subset) if subset else 0
        }
    
    # Group by object count
    for n_obj in set(r["n_objects"] for r in results):
        subset = [r for r in results if r["n_objects"] == n_obj]
        summary["by_n_objects"][n_obj] = {
            "total": len(subset),
            "successful": sum(1 for r in subset if r["success"]),
            "success_rate": sum(1 for r in subset if r["success"]) / len(subset) if subset else 0
        }
    
    # Save summary
    summary_file = os.path.join(batch_dir, "batch_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    return summary

# ============================================================================
# MAIN BATCH EXECUTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Batch experiment runner for OWG uncertainty research",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (2 experiments --default gpt-4o)
  python notebooks/batch_experiments.py --mode quick
  
  # Baseline vs uncertainty comparison (40 experiments)
  python notebooks/batch_experiments.py --mode baseline_vs_uncertainty
  
  # Model comparison GPT-4o vs GPT-4o-mini (40 experiments)
  python notebooks/batch_experiments.py --mode model_comparison --models gpt-4o gpt-4o-mini

  # Full factorial design (180 experiments)
  python notebooks/batch_experiments.py --mode full
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["quick", "baseline_vs_uncertainty", "model_comparison", "full", "custom"],
        default="baseline_vs_uncertainty",
        help="Experiment mode (default: baseline_vs_uncertainty)"
    )
    
    parser.add_argument(
        "--experiment-name",
        default="batch_exp",
        help="Name for this batch (default: batch_exp)"
    )
    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous incomplete batch"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout per experiment in seconds (default: 300)"
    )
    
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gpt-4o"],
        help="List of model IDs to compare"
    )

    args = parser.parse_args()
    
    # Create batch directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = f"experiments/{args.experiment_name}_{timestamp}"
    
    if args.resume:
        # Find most recent batch directory
        exp_dirs = sorted(Path("experiments").glob(f"{args.experiment_name}_*"))
        if exp_dirs:
            batch_dir = str(exp_dirs[-1])
            print(f"📂 Resuming from: {batch_dir}")
        else:
            print(f"⚠️  No previous batch found to resume. Starting fresh.")
            args.resume = False
    
    os.makedirs(batch_dir, exist_ok=True)
    
    # Define experiment matrix based on mode
    if args.mode == "quick":
        seeds = SEEDS[:2]
        queries = QUERIES[:1]
        n_objects = [12]
        prompt_types = ["baseline", "confidence"]
        models = args.models
        
    elif args.mode == "baseline_vs_uncertainty":
        seeds = SEEDS
        queries = QUERIES
        n_objects = [12]
        prompt_types = ["baseline", "confidence", "hedging", "cautious", "uncertainty_description"]
        models = args.models
        
    elif args.mode == "model_comparison":
        seeds = SEEDS
        queries = QUERIES
        n_objects = [12]
        prompt_types = ["baseline", "confidence"]
        models = args.models
        
    else:  # full
        seeds = SEEDS
        queries = QUERIES
        n_objects = N_OBJECTS_RANGE
        prompt_types = list(PROMPT_CONFIGS.keys())
        models = args.models
    
    # Generate all experiment combinations
    experiments = list(product(seeds, queries, n_objects, prompt_types, models))
    total_experiments = len(experiments)
    
    # Load completed experiments if resuming
    completed = load_completed_experiments(batch_dir) if args.resume else set()
    remaining = [exp for exp in experiments if exp not in completed]
    
    # Print batch information
    print("\n" + "="*70)
    print(f"🧪 BATCH EXPERIMENT: {args.experiment_name}")
    print("="*70)
    print(f"📊 Mode: {args.mode}")
    print(f"📁 Output directory: {batch_dir}")
    print(f"🎲 Seeds: {seeds}")
    print(f"💬 Queries: {len(queries)}")
    print(f"🎯 Object counts: {n_objects}")
    print(f"📝 Prompt types: {prompt_types}")
    print(f"🤖 Models: {models}")
    print(f"\n📈 Total experiments: {total_experiments}")
    
    if args.resume:
        print(f"✅ Already completed: {len(completed)}")
        print(f"⏳ Remaining: {len(remaining)}")
    
    print("="*70 + "\n")
    
    # Confirm before starting
    if len(remaining) > 10 and not args.resume:
        confirm = input(f"Start {len(remaining)} experiments? [y/N]: ")
        if confirm.lower() != 'y':
            print("Aborted.")
            return
    
    # Run experiments
    start_time = time.time()
    
    for idx, (seed, query, n_obj, prompt_type, model_name) in enumerate(remaining, 1):
        # Generate config for this experiment
        config_path = os.path.join(batch_dir, f"config_{prompt_type}_{model_name}_{seed}.yaml")
        
        if not os.path.exists(config_path):
            generate_config_file(
                PROMPT_CONFIGS[prompt_type],
                MODEL_CONFIGS[model_name],
                config_path
            )
        
        # Create output directory
        exp_output = os.path.join(
            batch_dir,
            f"run_{idx:03d}_s{seed}_n{n_obj}_{prompt_type}_{model_name}"
        )
        os.makedirs(exp_output, exist_ok=True)
        
        # Run experiment
        print(f"[{idx}/{len(remaining)}] ", end="")
        print(f"seed={seed}, query='{query[:30]}...', n_obj={n_obj}, prompt={prompt_type}, model={model_name}")
        
        result = run_single_experiment(seed, query, n_obj, config_path, exp_output, args.timeout)
        
        # Log result
        result_record = {
            "run_id": len(completed) + idx,
            "seed": seed,
            "query": query,
            "n_objects": n_obj,
            "prompt_type": prompt_type,
            "model_name": model_name,
            "model_category": MODEL_CONFIGS[model_name]["category"],
            "timestamp": datetime.now().isoformat(),
            "success": result["success"],
            "attempts": result.get("attempts", 1),
            "output_dir": exp_output,
            "config_path": config_path
        }
        
        if not result["success"]:
            result_record["error"] = result.get("error", result.get("stderr", "unknown"))
            print(f"   ❌ FAILED: {result_record.get('error', 'unknown')[:50]}")
        else:
            print(f"   ✅ SUCCESS (attempt {result['attempts']})")
        
        save_experiment_result(result_record, batch_dir)
        
        # Progress update every 10 experiments
        if idx % 10 == 0:
            elapsed = time.time() - start_time
            rate = idx / elapsed
            remaining_time = (len(remaining) - idx) / rate if rate > 0 else 0
            print(f"\n   ⏱️  Progress: {idx}/{len(remaining)} | "
                  f"Elapsed: {elapsed/60:.1f}m | "
                  f"ETA: {remaining_time/60:.1f}m\n")
    
    # Generate summary
    print("\n" + "="*70)
    print("GENERATING SUMMARY REPORT")
    print("="*70)
    
    summary = generate_summary_report(batch_dir)
    
    if summary:
        print(f"\n📊 BATCH COMPLETE")
        print(f"   Total: {summary['total_experiments']}")
        print(f"   ✅ Successful: {summary['successful']}")
        print(f"   ❌ Failed: {summary['failed']}")
        print(f"   Success rate: {summary['successful']/summary['total_experiments']*100:.1f}%")
        
        print(f"\n📝 By Prompt Type:")
        for ptype, stats in summary["by_prompt_type"].items():
            print(f"   {ptype}: {stats['successful']}/{stats['total']} "
                  f"({stats['success_rate']*100:.1f}%)")
        
        print(f"\n🤖 By Model:")
        for model, stats in summary["by_model"].items():
            print(f"   {model}: {stats['successful']}/{stats['total']} "
                  f"({stats['success_rate']*100:.1f}%)")
        
        print(f"\n🎯 By Object Count:")
        for n_obj, stats in sorted(summary["by_n_objects"].items()):
            print(f"   {n_obj} objects: {stats['successful']}/{stats['total']} "
                  f"({stats['success_rate']*100:.1f}%)")
    
    elapsed_total = time.time() - start_time
    print(f"\n⏱️  Total time: {elapsed_total/60:.1f} minutes")
    print(f"📁 Results saved to: {batch_dir}/")
    print(f"   - batch_results.jsonl (detailed logs)")
    print(f"   - batch_summary.json (aggregated stats)")
    print("="*70 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user. Progress has been saved.")
        print("   Resume with: --resume flag\n")
        sys.exit(0)