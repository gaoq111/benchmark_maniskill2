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

# Assuming the following imports are from custom modules

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

data_dir = "../benchmark/sequence/temporal_reordering"

# Clear and recreate the data directory
if os.path.exists(data_dir):
    shutil.rmtree(data_dir)
os.makedirs(data_dir)

# Environment and object parameters
objects = ['cube', 'sphere', 'custom']
custom_names = ['002_master_chef_can', '003_cracker_box', '004_sugar_box', '006_mustard_bottle', '012_strawberry', '013_apple']
object_sizes = [0.03, 0.04, 0.05, 0.06]
object_colors = list(color_maps.keys())
direction_length = 0.25
initial_position_ranges = [
    ((-0.35, 0.35), (-0.35, 0.35))
]

# Speed
speeds = [5, 10, 15, 20, 25]

# All camera angles, excluding side3
camera_angles = [f"{angle}{background}" for angle, background in product(['front', 'side', 'top'], [0, 1, 2, 3]) if not (angle == 'side' and background == 3)]

# Distractors
num_distractors = [0, 1, 2, 3]  # Possible numbers of distractor objects

# Initialize pair of trajectories
moving_directions = [["circle_lc"], ["circle_lcc"], ["circle_rc"], ["circle_rcc"], ['right'], ['left'], ['behind'], ['front']]

# Add sampling options
sampling_options = ['even']

# Add transparency options
transparency_options = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]  # 1.0 is fully opaque, 0.5 is half transparent

def generate_sequence(env, directions, speed_step, camera_angle):
    direction_configs = [(0, 0, direction) for direction in directions]
    distances = [direction_length for _ in directions]
    background = int(camera_angle[-1])  # Extract background number from camera_angle
    all_frames = env.move_in_directions(direction_configs, distances, steps=speed_step, camera_view="all", background=background)
    
    # The camera_angle (e.g., 'front1') is now directly usable as the key
    return all_frames[camera_angle]

def select_key_frames(all_frames, num_key_frames=5, sampling_method='random'):
    num_frames = len(all_frames)
    
    if sampling_method == 'even':
        # Even sampling
        step = num_frames // num_key_frames
        key_frame_indices = [i * step for i in range(num_key_frames)]
    else:
        # Random sampling (current method)
        key_frame_indices = sorted(random.sample(range(num_frames), num_key_frames))
    
    key_frames = [all_frames[i] for i in key_frame_indices]
    return key_frames, key_frame_indices

def create_task(env, task_id, variations):
    object, size, color, ranges, speed_step, sampling_method, transparency, camera_angle, num_distractors, custom_name = variations
    
    # Extract background from camera_angle
    background = int(camera_angle[-1])
    camera_angle_type = camera_angle[:-1]  # 'front', 'side', or 'top'

    # If background 2 make range small:
    if background == 2:
        ranges = ((-0.2, 0.2), (-0.1, 0.1))

    # Create main object
    if object == 'custom':
        obj_config = {'obj_type': object, 'name': custom_name, 'size': size*1.2, 'ranges': ranges, 'color': ['predefinedColor'], 'static': True}
    else:
        obj_config = {
            'obj_type': object, 
            'name': "obj_1", 
            'size': size, 
            'ranges': ranges,
            'color': (*color_maps[color], transparency),
            'static': True
        }
    
    # Create distractor objects
    distractor_configs = []
    for i in range(num_distractors):
        distractor_objects = ['sphere', 'cube']
        obj_choice = random.choice(distractor_objects)
        
        distractor_config = {
            'obj_type': obj_choice,
            'name': f"distractor_{i}",
            'size': random.choice(object_sizes),
            'ranges': ranges,
            'color': (*color_maps[random.choice(object_colors)], random.choice(transparency_options)),
            'static': True
        }

        distractor_configs.append(distractor_config)
    
    # Combine main object config with distractor configs
    all_configs = [obj_config] + distractor_configs
    
    configs_collate = collate_infos(all_configs)
    env.register_configures(configs_collate)
    _ = env.reset(options={"reconfigure": True})
    env.initialize_objects(background=background)
    
    directions = random.choice(moving_directions)
    
    all_frames = generate_sequence(env, directions, speed_step, camera_angle)
    
    # Select 5 key frames using the new function
    key_frames, key_frame_indices = select_key_frames(all_frames, num_key_frames=5, sampling_method=sampling_method)
    
    # Shuffle the key frames
    shuffled_indices = list(range(5))
    random.shuffle(shuffled_indices)
    
    correct_order = [shuffled_indices.index(i) for i in range(5)]
    
    return key_frames, correct_order, shuffled_indices, directions

def generate_answer_choices(correct_order):
    choices = [correct_order.copy()]
    while len(choices) < 4:
        new_choice = correct_order.copy()
        random.shuffle(new_choice)
        if new_choice not in choices:
            choices.append(new_choice)
    random.shuffle(choices)
    return choices

def save_frames_to_task_dir(frames, task_dir, shuffled_indices):
    for i, frame_index in enumerate(shuffled_indices):
        frame = frames[frame_index]
        frame_path = os.path.join(task_dir, f"frame_{i}.png")
        plt.imsave(frame_path, frame)

def create_full_sequence_video(task_dir, correct_order, fps=5):
    # Get all frame files in the task directory
    frame_files = [f"frame_{i}.png" for i in range(5)]  # We know there are 5 frames
    
    # Read frames in the correct order
    frames = []
    for i in correct_order:
        img_path = os.path.join(task_dir, frame_files[i])
        img = Image.open(img_path)
        frames.append(np.array(img))

    video_name = os.path.join(task_dir, "ordered_sequence.mp4")
    generate_video(video_name, frames, fps=fps, color_change=False)
    return video_name

def update_index(task_id, randomized_order, correct_order, choices, variations, directions):
    index_path = os.path.join(data_dir, "index.jsonl")
    object, size, color, ranges, speed_step, sampling_method, transparency, camera_angle, num_distractors, custom_name = variations

    if object != 'custom':
        custom_name = "N/A"

    background = int(camera_angle[-1])
    camera_angle_type = camera_angle[:-1]
    with jsonlines.open(index_path, mode='a') as writer:
        writer.write({
            "task_id": task_id,
            "randomized_order": randomized_order,
            "correct_order": correct_order,
            "choices": choices,
            "object": object,
            "custom_name": custom_name,
            "object_size": size,
            "object_color": color,
            "initial_position_range": ranges,
            "speed": speed_step,
            "movement_directions": directions,
            "sampling_method": sampling_method,
            "transparency": transparency,
            "camera_angle": camera_angle,
            "camera_angle_type": camera_angle_type,
            "background": background,
            "num_distractor_objects": num_distractors,
        })

def process_all_tasks(data_dir, fps=5):
    video_paths = []
    index_path = os.path.join(data_dir, "index.jsonl")
    with open(index_path, 'r') as f:
        for line in f:
            task_data = json.loads(line)
            task_id = task_data['task_id']
            correct_order = task_data['correct_order']
            task_dir = os.path.join(data_dir, f"task_{task_id}")
            video_path = create_full_sequence_video(task_dir, correct_order, fps)
            video_paths.append(video_path)
    return video_paths

# Main execution
num_tasks = 50
env = gym.make("CustomEnv-v0", obs_mode="rgbd")

all_variations = list(product(
    objects, object_sizes, object_colors, initial_position_ranges, speeds, 
    sampling_options, transparency_options, camera_angles, num_distractors, custom_names
))

random.shuffle(all_variations)
selected_variations = all_variations[:num_tasks]

task_id = 0
for variations in selected_variations:
    key_frames, correct_order, shuffled_indices, directions = create_task(env, task_id, variations)
    choices = generate_answer_choices(correct_order)
    
    task_dir = os.path.join(data_dir, f"task_{task_id}")
    os.makedirs(task_dir, exist_ok=True)
    
    save_frames_to_task_dir(key_frames, task_dir, shuffled_indices)
    
    update_index(task_id, shuffled_indices, correct_order, choices, variations, directions)
    
    task_id += 1
    
    if task_id >= num_tasks:
        break

env.close()

print(f"Generated {task_id} Sequence Reordering tasks in {data_dir}")
print(f"Index file created at {os.path.join(data_dir, 'index.jsonl')}")

# Create videos
video_paths = process_all_tasks(data_dir)
print(f"Created {len(video_paths)} ordered sequence videos in their respective task folders")