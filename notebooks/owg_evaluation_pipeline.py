#!/usr/bin/env python3
"""
OWG Evaluation Pipeline - Command Line Script
Author: Vanessa (converted to CLI by Assistant)

Usage:
    python owg_evaluation_pipeline.py --seed 42 --config config/pyb/OWG_mod.yaml --query "Remove the smallest object"
"""

import argparse
import sys
import os
import time
import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for CLI
import matplotlib.pyplot as plt
from datetime import datetime

# Add parent directory to path to import owg modules
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Add third_party directory to path
third_party_path = os.path.join(parent_dir, 'third_party', 'grconvnet')
if third_party_path not in sys.path:
    sys.path.append(third_party_path)

# Import project modules
from owg_robot.env import *
from owg_robot.camera import Camera
from owg_robot.objects import YcbObjects
from third_party.grconvnet import *
from owg.utils.grasp import Grasp2D
from owg.visual_prompt import VisualPrompterGrounding, VisualPrompterPlanning, VisualPrompterGraspRanking
from owg_mod.tracker import GraspStatsTracker


def display_image(path_or_array, size=(10, 10), save_path=None):
    """Helper function to display/save images"""
    if isinstance(path_or_array, str):
        image = np.asarray(Image.open(open(path_or_array, 'rb')).convert("RGB"))
    else:
        image = path_or_array
    
    plt.figure(figsize=size)
    plt.imshow(image)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        print(f"Image saved to {save_path}")
    plt.close()


def setup_environment(seed, n_objects=12):
    """Setup PyBullet environment with objects"""
    print("Setting up PyBullet environment...")
    
    # Load camera and env
    center_x, center_y, center_z = CAM_X, CAM_Y, CAM_Z
    camera = Camera((center_x, center_y, center_z), (center_x, center_y, 0.785), 
                   0.2, 2.0, (448, 448), 40)
    env = Environment(camera, vis=True, asset_root='./owg_robot/assets', 
                     debug=False, finger_length=0.06)
    
    # Load objects
    objects = YcbObjects('./owg_robot/assets/ycb_objects',
                        mod_orn=['ChipsCan', 'MustardBottle', 'TomatoSoupCan'],
                        mod_stiffness=['Strawberry'],
                        seed=seed)
    objects.shuffle_objects()
    
    for obj_name in objects.obj_names[:n_objects]:
        path, mod_orn, mod_stiffness = objects.get_obj_info(obj_name)
        env.load_isolated_obj(path, obj_name, mod_orn, mod_stiffness)
    env.dummy_simulation_steps(10)
    
    print(f"Loaded {n_objects} objects with seed {seed}")
    return env, camera


def setup_grasps(env, grasp_generator, camera, visualise_grasps=False):
    """Setup grasps for all objects in environment"""
    print("Generating grasps for objects...")
    
    rgb, depth, seg = env.camera.get_cam_img()    
    img_size = grasp_generator.IMG_WIDTH
    
    if img_size != camera.width: 
        rgb = cv2.resize(rgb, (img_size, img_size))
        depth = cv2.resize(depth, (img_size, img_size))
    
    for obj_id in env.obj_ids:
        mask = seg == obj_id
        if img_size != camera.width:
            mask = np.array(Image.fromarray(mask).resize((img_size, img_size), Image.LANCZOS))
        
        grasps, grasp_rects = grasp_generator.predict_grasp_from_mask(
            rgb, depth, mask, n_grasps=5, show_output=False)
        
        if img_size != camera.width:
            # Normalize to original size
            for j, gr in enumerate(grasp_rects):
                grasp_rects[j][0] = int(gr[0] / img_size * camera.width)
                grasp_rects[j][1] = int(gr[1] / img_size * camera.width)
                grasp_rects[j][4] = int(gr[4] / img_size * camera.width)
                grasp_rects[j][3] = int(gr[3] / img_size * camera.width)
        
        grasp_rects = [Grasp2D.from_vector(
            x=g[1], y=g[0], w=g[4], h=g[3], theta=g[2], 
            W=camera.width, H=camera.width, normalized=False, line_offset=5,
        ) for g in grasp_rects]
        
        env.set_obj_grasps(obj_id, grasps, grasp_rects)
    
    if visualise_grasps:
        LID = []
        for obj_id in env.obj_ids:
            grasps = env.get_obj_grasps(obj_id)
            color = np.random.rand(3).tolist()
            for g in grasps:
                LID = env.draw_predicted_grasp(g, color=color, lineIDs=LID)
        
        time.sleep(1)
        env.remove_drawing(LID)
        env.dummy_simulation_steps(10)
    
    print(f"Generated grasps for {len(env.obj_ids)} objects")


def execute_and_track_actions(actions, env, image, all_masks, all_grasp_rects, 
                              obj_ids, grasp_ranker, tracker, output_dir):
    """Execute grasping actions and track results"""
    if isinstance(actions, dict):
        actions = [actions]
    
    os.makedirs(output_dir, exist_ok=True)
    
    for idx, act in enumerate(actions):
        print(f"\n--- Executing action {idx + 1}/{len(actions)} ---")
        print(f"Action: {act['action']}, Object ID: {act['input']}")
        
        obj_id = act['input']
        obj_grasps = all_grasp_rects[obj_id]
        obj_mask = all_masks[np.where(obj_ids == obj_id)[0][0]]
        req_data = {'grasps': obj_grasps, 'mask': obj_mask}
        
        # Save visual prompt
        visual_prompt, _ = grasp_ranker.prepare_image_prompt(image.copy(), req_data)
        marked_image_grasping = visual_prompt[-1]
        display_image(marked_image_grasping, (12, 6), 
                     save_path=os.path.join(output_dir, f'grasp_visual_{idx}.png'))
        
        # Rank grasps
        sorted_grasps, best_grasp, sorted_grasp_indices, metadata_rank = \
            grasp_ranker.request(image.copy(), req_data)
        act['grasps'] = sorted_grasp_indices
        
        # Execute grasp
        if act['action'] == 'remove':
            success_grasp, success_target, num_attempts = \
                env.put_obj_in_free_space(obj_id, grasp_indices=act['grasps'])
        elif act['action'] == 'pick':
            success_grasp, success_target, num_attempts = \
                env.put_obj_in_tray(obj_id, grasp_indices=act['grasps'])
        else:
            print(f"Unknown action type: {act['action']}")
            continue
        
        for _ in range(30):
            env.step_simulation()
        
        # Log result
        tracker.record_grasp(
            success=success_grasp,
            object_id=obj_id,
            position=env.get_obj_pos(obj_id),
            retries=num_attempts - 1,
            grasp_index=act['grasps'],
            additional_info={"timestamp": datetime.now().isoformat()}
        )
        
        tracker.set_metadata(metadata_rank, module_name="ranker")
        
        print(f"Success: {success_grasp}, Attempts: {num_attempts}")


def main():
    parser = argparse.ArgumentParser(
        description='OWG Evaluation Pipeline - Robotic Grasping with VLM',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for YcbObjects')
    parser.add_argument('--config', type=str, default='config/pyb/OWG_mod.yaml',
                       help='Path to configuration file')
    parser.add_argument('--query', type=str, default='Remove the smallest object',
                       help='User query for object selection')
    parser.add_argument('--n-objects', type=int, default=12,
                       help='Number of objects to load')
    parser.add_argument('--output-dir', type=str, default='output',
                       help='Directory to save output visualizations')
    parser.add_argument('--visualise-grasps', action='store_true',
                       help='Visualize grasps in PyBullet GUI')
    parser.add_argument('--openai-key', type=str, default='test',
                       help='OpenAI API key (if needed)')
    
    args = parser.parse_args()
    
    # Set OpenAI API key
    os.environ['OPENAI_API_KEY'] = args.openai_key
    
    print("=" * 60)
    print("OWG Evaluation Pipeline")
    print("=" * 60)
    print(f"Seed: {args.seed}")
    print(f"Config: {args.config}")
    print(f"Query: {args.query}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)
    
    # Setup environment
    env, camera = setup_environment(args.seed, args.n_objects)
    
    # Load grasp generator
    print("\nLoading grasp generator...")
    grasp_generator = load_grasp_generator(camera)
    
    # Setup grasps
    setup_grasps(env, grasp_generator, camera, args.visualise_grasps)
    
    # Get observations
    obs = env.get_obs()
    all_grasp_rects = {k: env.get_obj_grasp_rects(k) for k in env.obj_ids}
    
    # Prepare segmentation data
    image, seg = obs['image'], obs['seg']
    obj_ids = np.unique(seg)[1:]
    all_masks = np.stack([seg == objID for objID in obj_ids])
    marker_data = {'masks': all_masks, 'labels': obj_ids}
    
    # Initialize modules
    print("\nInitializing VLM modules...")
    grounder = VisualPrompterGrounding(args.config, debug=False)
    planner = VisualPrompterPlanning(args.config, debug=False)
    grasp_ranker = VisualPrompterGraspRanking(args.config, debug=False)
    
    # Initialize tracker
    tracker = GraspStatsTracker()
    
    # Referring segmentation
    print(f"\n--- Grounding Query: '{args.query}' ---")
    visual_prompt, _ = grounder.prepare_image_prompt(image.copy(), marker_data)
    marked_image_grounding = visual_prompt[-1]
    os.makedirs(args.output_dir, exist_ok=True)
    display_image(marked_image_grounding, (6, 6), 
                 save_path=os.path.join(args.output_dir, 'grounding_visual.png'))
    
    dets, target_mask, target_ids, metadata_ground = grounder.request(
        text_query=args.query, image=image.copy(), data=marker_data)
    
    target_id = target_ids[0]
    print(f"Target object ID: {target_id}")
    display_image(target_mask, (6, 6), 
                 save_path=os.path.join(args.output_dir, 'target_mask.png'))
    
    # Grasp planning
    print("\n--- Planning Actions ---")
    plan, metadata_plan = planner.request(
        text_query=target_id, image=image.copy(), data=marker_data)
    action = plan
    print(f"Planned action: {action}")
    
    # Execute actions
    print("\n--- Executing Actions ---")
    execute_and_track_actions(
        actions=action,
        env=env,
        image=image,
        all_masks=all_masks,
        all_grasp_rects=all_grasp_rects,
        obj_ids=obj_ids,
        grasp_ranker=grasp_ranker,
        tracker=tracker,
        output_dir=args.output_dir
    )
    
    # Save tracking results
    print("\n--- Experiment Tracking ---")
    tracker.set_metadata(metadata_ground, module_name="grounder")
    tracker.set_metadata(metadata_plan, module_name="planner")
    
    tracker.set_model_settings({
        "grounder": grounder.get_model_params(),
        "ranker": grasp_ranker.get_model_params(),
        "planner": planner.get_model_params()
    })
    
    tracker.set_prompt_variants({
        "grounder": grounder.get_variants(),
        "ranker": grasp_ranker.get_variants(),
        "planner": planner.get_variants()
    })
    
    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Overall success rate: {tracker.get_success_rate():.2%}")
    print(f"\nGrasp log:")
    for log_entry in tracker.get_log():
        print(f"  {log_entry}")
    
    summary = tracker.get_summary()
    print(f"\nExperiment summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    tracker.save_uncertainty_log(summary)
    print("\n✅ Uncertainty log saved to logs/uncertainty_logs.jsonl")
    print(f"✅ Visualizations saved to {args.output_dir}/")
    print("=" * 60)
    
    # Success rate per object
    per_obj_success = tracker.get_success_rate_per_object()
    if per_obj_success:
        print("\nSuccess rate per object:")
        for obj_id, rate in per_obj_success.items():
            print(f"  Object {obj_id}: {rate:.2%}")
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)