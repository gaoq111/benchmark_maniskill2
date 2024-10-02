import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
import sys
sys.path.append('..')
from PIL import Image
from tqdm import trange
from IPython.display import clear_output
from sapien.core import Pose
from mani_skill2.envs.pick_and_place.base_env import StationaryManipulationEnv
from mani_skill2.utils.registration import register_env
import gymnasium as gym
import sapien.core as sapien
from mani_skill2 import ASSET_DIR
from pathlib import Path
from typing import Dict, List
from mani_skill2 import format_path
from mani_skill2.utils.io_utils import load_json
import numpy as np
from sapien.core import Pose
import matplotlib.pyplot as plt
from mani_skill2.agents.base_agent import BaseAgent
import os
import mani_skill2.envs
from mani_skill2 import ASSET_DIR
from mani_skill2.utils.sapien_utils import look_at
from mani_skill2.utils.registration import register_env
from mani_skill2.envs.pick_and_place.pick_cube import PickCubeEnv
from mani_skill2.utils.sapien_utils import hex2rgba
from mani_skill2.sensors.camera import CameraConfig
from transforms3d.euler import euler2quat, quat2euler
from mani_skill2.utils.sapien_utils import vectorize_pose
from mani_skill2.agents.base_controller import BaseController
from IPython.display import Video
import shutil
import random
import json
import jsonlines
from itertools import product

from utils import *
from agent_env_utils import *
from plot_utils import *

color_maps = {
    "red": (1, 0, 0),
    "blue": (0, 0, 1),
    "green": (0, 1, 0),
    "white": (1, 1, 1),
    "black": (0, 0, 0),
    "yellow": (1, 1, 0),
    "orange": (1, 0.5, 0),
    "purple": (0.5, 0, 0.5),
    "gray": (0.5, 0.5, 0.5),
}
directions = ["front", "behind", "right", "left", "top"]

data_dir = "../benchmark/sequence/temporal_extension"

if os.path.exists(data_dir):
    shutil.rmtree(data_dir)
os.makedirs(data_dir)

objects = ['sphere', 'cube', 'custom']
custom_names = ['002_master_chef_can', '003_cracker_box', '004_sugar_box', '006_mustard_bottle', '012_strawberry', '013_apple']
object_sizes = [0.03, 0.04, 0.05, 0.06]
object_colors = list(color_maps.keys())
direction_length = 0.25
initial_position_ranges = [
    ((-0.35, 0.35), (-0.35, 0.35)),
    ((-0.2, 0.2), (-0.2, 0.2)),
]

speeds = [5, 6, 7, 8, 9, 10]

camera_angles = [f"{angle}{background}" for angle, background in product(['front', 'side', 'top'], [0, 1, 2, 3]) if not (angle == 'side' and background == 3)]

num_distractors = [0]

moving_directions = [["circle_lc"], ["circle_lcc"], ["circle_rc"], ["circle_rcc"], ['right'], ['left'], ['behind'], ['front']]

sampling_options = ['even']

transparency_options = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]

movement_ranges = [
    ((0, 20), (21, 40)),  # Object A moves from 0-20, then Object B moves from 21-40
    ((21, 40), (0, 20)),  # Object B moves from 0-20, then Object A moves from 21-40
    ((0, 20), (21, 40)),  # Object A moves from 0-20, then Object B moves from 21-40
    ((0, 15), (16, 40)),  # Object A moves from 0-15, then Object B moves from 16-40
    ((0, 20), (21, 40)),  # Object A moves from 0-20, then Object B moves from 21-40
    ((21, 40), (0, 20))  # Object B moves from 0-20, then Object A moves from 21-40
]

def generate_sequence(env, directions_list, camera_angle, object_A_range, object_B_range, total_frames=40):
    distances = [direction_length, direction_length]
    all_frames = []

    # Move Object A
    if object_A_range[1] > object_A_range[0]:
        frames_A = env.move_in_directions(
            [(0, 0, directions_list[0][0])], 
            [distances[0]], 
            steps=object_A_range[1] - object_A_range[0], 
            camera_view=camera_angle
        )
        all_frames.extend(frames_A)
    
    # Calculate frames where Object A is stationary
    stationary_A_frames = object_B_range[0] - object_A_range[1]
    if stationary_A_frames > 0:
        last_A_frame = all_frames[-1] if all_frames else env.render(camera_angle)
        all_frames.extend([last_A_frame] * stationary_A_frames)

    # Move Object B
    if object_B_range[1] > object_B_range[0]:
        frames_B = env.move_in_directions(
            [(1, 1, directions_list[1][0])], 
            [distances[1]], 
            steps=object_B_range[1] - object_B_range[0], 
            camera_view=camera_angle
        )
        all_frames.extend(frames_B)

    # Ensure we have exactly total_frames
    if len(all_frames) < total_frames:
        all_frames.extend([all_frames[-1]] * (total_frames - len(all_frames)))
    elif len(all_frames) > total_frames:
        all_frames = all_frames[:total_frames]

    # Select 5 evenly spaced frames
    selected_indices = np.linspace(0, len(all_frames) - 1, 5, dtype=int)
    selected_frames = [all_frames[i] for i in selected_indices]
    
    return selected_frames, selected_indices, all_frames

def create_task(env, task_id, variations):
    object1, object2, size1, color1, size2, color2, ranges, speed_step, sampling_method, transparency, camera_angle, num_distractors, custom_name1, custom_name2 = variations
    
    background = int(camera_angle[-1])
    
    # Configure first object
    if object1 == 'custom':
        obj_config_1 = {'obj_type': 'custom', 'name': custom_name1, 'size': size1*1.2, 'ranges': ranges, 'color': ['predefinedColor'], 'static': True}
    else:
        obj_config_1 = {'obj_type': object1, 'name': "obj_1", 'size': size1, 'ranges': ranges, 'color': (*color_maps[color1], transparency), 'static': True}
    
    # Configure second object
    if object2 == 'custom':
        obj_config_2 = {'obj_type': 'custom', 'name': custom_name2, 'size': size2*1.2, 'ranges': ranges, 'color': ['predefinedColor'], 'static': True}
    else:
        obj_config_2 = {'obj_type': object2, 'name': "obj_2", 'size': size2, 'ranges': ranges, 'color': (*color_maps[color2], transparency), 'static': True}
    
    all_configs = [obj_config_1, obj_config_2]
    
    configs_collate = collate_infos(all_configs)
    env.register_configures(configs_collate)
    _ = env.reset(options={"reconfigure": True})
    env.initialize_objects(background=background)
    
    directions_1 = random.choice(moving_directions)
    directions_2 = random.choice(moving_directions)
    
    # Randomly select a movement range
    object_A_range, object_B_range = random.choice(movement_ranges)
    
    selected_frames, selected_indices, all_frames = generate_sequence(env, [directions_1, directions_2], camera_angle, object_A_range, object_B_range)
    
    return selected_frames, directions_1, directions_2, object_A_range, object_B_range, selected_indices, all_frames

def save_frames_to_task_dir(frames, task_dir):
    for i, frame in enumerate(frames):
        frame_dir = os.path.join(task_dir, f"frame_{i}")
        save_images_pararell([frame], frame_dir)
        
        src_path = os.path.join(frame_dir, "image_0.png")
        dst_path = os.path.join(task_dir, f"frame_{i}.png")
        shutil.move(src_path, dst_path)
        
        os.rmdir(frame_dir)

def create_ordered_video(task_dir, fps=5):
    ordered_images = []
    for i in range(5):
        img_path = os.path.join(task_dir, f"frame_{i}.png")
        img = Image.open(img_path)
        ordered_images.append(np.array(img))

    video_name = os.path.join(task_dir, "ordered_sequence.mp4")
    generate_video(video_name, ordered_images, fps=fps, color_change=False)
    return video_name

def process_all_tasks(data_dir, fps=5):
    video_paths = []
    index_path = os.path.join(data_dir, "index.jsonl")
    with open(index_path, 'r') as f:
        for line in f:
            task_data = json.loads(line)
            task_id = task_data['task_id']
            task_dir = os.path.join(data_dir, f"task_{task_id}")
            video_path = create_ordered_video(task_dir, fps)
            video_paths.append(video_path)
    return video_paths


def create_full_video(all_frames, task_dir, fps=10):
    """
    Creates a video using all frames from the simulation.

    Args:
    all_frames (list): A list of numpy arrays, each representing a frame.
    task_dir (str): The directory where the video will be saved.
    fps (int): Frames per second for the video. Default is 10.

    Returns:
    str: The path to the created video file.
    """
    # Ensure all frames are numpy arrays
    all_frames = [np.array(frame) if not isinstance(frame, np.ndarray) else frame for frame in all_frames]

    # Create the video file name
    video_name = os.path.join(task_dir, "full_sequence.mp4")

    # Generate the video
    generate_video(video_name, all_frames, fps=fps, color_change=False)

    return video_name

def update_index(task_id, variations, directions_1, directions_2, object_A_range, object_B_range, selected_indices):
    index_path = os.path.join(data_dir, "index.jsonl")
    object1, object2, size1, color1, size2, color2, ranges, speed_step, sampling_method, transparency, camera_angle, num_distractors, custom_name1, custom_name2 = variations

    if object1 == 'custom' and object2 == 'custom':
        object_1_description = f"the {custom_name1}"
        object_2_description = f"the {custom_name2}"
    elif object1 == "custom" and object2 != "custom":
        object_1_description = f"the {custom_name1}"
        object_2_description = f"the {color2} {object2}"
    elif object1 != "custom" and object2 == 'custom':
        object_1_description = f"the {color1} {object1}"
        object_2_description = f"the {custom_name2}"
    else:
        object_1_description = f"the {color1} {object1}"
        object_2_description = f"the {color2} {object2}"



    if (object_A_range[1] - object_A_range[0]) - (object_B_range[1] - object_B_range[0]) > 1:
        correct_answer = object_1_description
    elif (object_A_range[1] - object_A_range[0]) - (object_B_range[1]-  object_B_range[0]) < -1:
        correct_answer = object_2_description
    else:
        correct_answer = "same time"

    with jsonlines.open(index_path, mode='a') as writer:
        writer.write({
            # Task metadata
            "task_id": task_id,
            "correct_answer": correct_answer,
            "multiple_choice_options": [
                object_1_description,
                object_2_description,
                "same duration"
            ],
            
            # Object 1 properties
            "object1_type": object1,
            "object_1_size": size1,
            "object_1_color": color1 if object1 != 'custom' else 'N/A',
            "custom_name1": custom_name1 if object1 == 'custom' else 'N/A',
            "object_1_directions": directions_1,
            "object_A_movement_range": object_A_range,
            
            # Object 2 properties
            "object2_type": object2,
            "object_2_size": size2,
            "object_2_color": color2 if object2 != 'custom' else 'N/A',
            "custom_name2": custom_name2 if object2 == 'custom' else 'N/A',
            "object_2_directions": directions_2,
            "object_B_movement_range": object_B_range,
            
            # Environment settings
            "initial_position_range": ranges,
            "speed": speed_step,
            "sampling_method": sampling_method,
            "transparency": transparency,
            "camera_angle": camera_angle,
            "num_distractor_objects": num_distractors
        })
# Main execution
num_tasks = 10
env = gym.make("CustomEnv-v0", obs_mode="rgbd")

#all_variations = list(product(
 #   objects, objects, object_sizes, object_colors, object_sizes, object_colors, initial_position_ranges, speeds, 
  #  sampling_options, transparency_options, camera_angles, num_distractors, custom_names, custom_names
#))

#random.shuffle(all_variations)
#selected_variations = all_variations[:num_tasks]



selected_variations = [
    (
        random.choice(['sphere', 'cube', 'custom']),
        random.choice(['sphere', 'cube', 'custom']),
        random.choice([0.03, 0.04, 0.05, 0.06]),
        random.choice(list(color_maps.keys())),
        random.choice([0.03, 0.04, 0.05, 0.06]),
        random.choice(list(color_maps.keys())),
        random.choice([((-0.35, 0.35), (-0.35, 0.35)), ((-0.2, 0.2), (-0.2, 0.2))]),
        random.choice([5, 6, 7, 8, 9, 10]),
        'even',
        random.choice([1.0, 0.9, 0.8, 0.7, 0.6, 0.5]),
        random.choice([f"{angle}{background}" for angle in ['front', 'side', 'top'] for background in [0, 1, 2, 3] if not (angle == 'side' and background == 3)]),
        0,
        random.choice(['002_master_chef_can', '003_cracker_box', '004_sugar_box', '006_mustard_bottle', '012_strawberry', '013_apple']),
        random.choice(['002_master_chef_can', '003_cracker_box', '004_sugar_box', '006_mustard_bottle', '012_strawberry', '013_apple'])
    )
    for _ in range(10)
]



task_id = 0
for variations in selected_variations:
    frames, directions_1, directions_2, object_A_range, object_B_range, selected_indices, all_frames = create_task(env, task_id, variations)
    
    task_dir = os.path.join(data_dir, f"task_{task_id}")
    os.makedirs(task_dir, exist_ok=True)
    
    save_frames_to_task_dir(frames, task_dir)

    full_video_path = create_full_video(all_frames, task_dir)
    
    update_index(task_id, variations, directions_1, directions_2, object_A_range, object_B_range, selected_indices)

    
    
    task_id += 1
    
    if task_id >= num_tasks:
        break

env.close()

#video_paths = process_all_tasks(data_dir)

print(type(all_frames[0]))

print(f"Generated {task_id} Temporal Order tasks in {data_dir}")
print(f"Index file created at {os.path.join(data_dir, 'index.jsonl')}")
