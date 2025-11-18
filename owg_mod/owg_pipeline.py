import numpy as np
import pybullet as p
import os
os.environ['OPENAI_API_KEY'] = 'sT3BlbkFJiteG5cfyk26lWKKaoVK2ImBcRr2MtDcB1UB9mB9UYpFqbm2RnIpjsTs7AK8qYCNh4-tl_CLcEA' 
from owg.visual_prompt import VisualPrompterGrounding, VisualPrompterPlanning, VisualPrompterGraspRanking
from owg.utils.grasp import Grasp2D
from owg_robot.env import *
from owg_robot.camera import Camera
from owg_robot.objects import YcbObjects
from owg_mod.tracker import GraspStatsTracker
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import FancyBboxPatch
from pprint import pprint
import sys
sys.path.append('/home/riche/owg_project/OWG/third_party/grconvnet')

def display_image(path_or_array, size=(10, 10)):
  if isinstance(path_or_array, str):
    image = np.asarray(Image.open(open(image_path, 'rb')).convert("RGB"))
  else:
    image = path_or_array
  
  plt.figure(figsize=size)
  plt.imshow(image)
  plt.axis('off')
  plt.show()

def init_simulation_environment(n_objects=12):
    camera = Camera(width=640, height=480, fov=60, near=0.01, far=10.0)
    env = Environment(camera, gui=True, debug=True)
    objects = YcbObjects(scale_range=(1.0, 1.0))
    objects.shuffle_objects()

    for obj_name in objects.obj_names[:n_objects]:
        obj_id = env.load_isolated_obj(obj_name)
        env.add_graspable_obj_id(obj_id)

    env.dummy_simulation_steps(10)
    return env, camera, objects


def setup_grasps(env, grasp_generator, visualise_grasps=True):
    obs = env.get_obs()
    image = obs['image']
    seg = obs['seg']
    all_ids = np.unique(seg)[1:]

    for obj_id in all_ids:
        grasp_poses, _, _ = grasp_generator(image, seg, obj_id, visualize=visualise_grasps)
        env.set_obj_grasp_poses(obj_id, grasp_poses)


def init_llm_modules(config_path):
    grounder = VisualPrompterGrounding(config_path, debug=True)
    planner = VisualPrompterPlanning(config_path, debug=True)
    ranker = VisualPrompterGraspRanking(config_path, debug=True)
    return grounder, planner, ranker


def run_grounding(grounder, image, marker_data, query):
    visual_prompt, _ = grounder.prepare_image_prompt(image.copy(), marker_data)
    marked_image = visual_prompt[-1]
    display_image(marked_image, (6, 6))
    return grounder.request(query, image.copy(), marker_data)


def run_planning(planner, target_id, image, marker_data):
    return planner.request(text_query=target_id, image=image.copy(), data=marker_data)


def run_ranking_and_execute(plan, env, image, all_masks, all_grasp_rects, obj_ids, ranker, tracker):
    for action in plan:
        id_ = action['object']
        query = action['query']
        if id_ not in obj_ids:
            continue

        grasps = all_grasp_rects.get(id_, [])
        if not grasps:
            continue

        masks = all_masks[obj_ids.tolist().index(id_)]
        best = ranker.request(text_query=query, image=image.copy(), masks=[masks], rects=[grasps])

        if best['success']:
            env.highlight_obj(id_)
            success = env.execute_grasp_rect(id_, grasps[best['grasp']['index']])
            tracker.update(success, object_id=id_, query=query)


def finalize_experiment(tracker, metadata_ground, metadata_plan, grounder, planner, ranker):
    tracker.set_metadata({"grounding": metadata_ground, "planning": metadata_plan})
    tracker.set_model_settings({
        "grounder": grounder.get_model_params(),
        "ranker": ranker.get_model_params(),
        "planner": planner.get_model_params()
    })
    tracker.set_prompt_variants({
        "grounder": grounder.get_variants(),
        "ranker": ranker.get_variants(),
        "planner": planner.get_variants()
    })
    print("Overall success rate:", tracker.get_success_rate())
    print("Grasp log:", tracker.get_log())
    print("Experiment summary:", tracker.get_summary())


def main():
    env, camera, _ = init_simulation_environment()
    grasp_generator = load_grasp_generator(camera)
    setup_grasps(env, grasp_generator, visualise_grasps=True)

    obs = env.get_obs()
    image, seg = obs['image'], obs['seg']
    obj_ids = np.unique(seg)[1:]
    all_masks = np.stack([seg == objID for objID in obj_ids])
    marker_data = {'masks': all_masks, 'labels': obj_ids}
    all_grasp_rects = {k: env.get_obj_grasp_rects(k) for k in env.obj_ids}

    grounder, planner, ranker = init_llm_modules('config/pyb/OWG_mod.yaml')
    dets, target_mask, target_ids, metadata_ground = run_grounding(grounder, image, marker_data, "I want to play tennis")
    plan, metadata_plan = run_planning(planner, target_ids[0], image, marker_data)

    tracker = GraspStatsTracker()
    run_ranking_and_execute(plan, env, image, all_masks, all_grasp_rects, obj_ids, ranker, tracker)
    finalize_experiment(tracker, metadata_ground, metadata_plan, grounder, planner, ranker)

    p.disconnect()


if __name__ == "__main__":
    main()
