import random
import itertools

# General settings
random_seed = 42
random.seed(random_seed)

# Directory settings
data_dir = "benchmark/motion/circle_full_ver3"
initial_dir = f"{data_dir}/initial"
final_dir = f"{data_dir}/final"

# Object settings
size_types = [1, 3]
space_types = ["cube-bound", "cube", "sphere", "sphere-bound"]
bound_types = ["front", "top"]
potential_sizes = [0.03, 0.05]

# Motion settings
direction_length = 0.2
scales = [1.1, 1.4]
moving_directions = [
    ["circle_lc"], 
    ["circle_lcc"],
    ["right", "behind", "left", "front"],
    ["right", "behind_left", "front_left"],
    ["right", "front", "behind", "right"]
]
moving_directions = moving_directions[:2]

# Color settings
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


# Background settings
background_quadrants = {
    0: [(-0.35, 0.6), (-0.35, -0.6), (0.35, -0.6), (0.35, 0.6)],
    1: [(-0.35, 0.5), (-0.35, -0.5), (0.3, -0.5), (0.3, 0.5)],
    2: [(0.12, -0.6), (0.12, 0.6), (-0.2, 0.6), (-0.2, -0.6)],
    3: [(1, 1), (-1, 1), (-1, -1), (1, -1)]
}
background_specs = {
    0: [(-0.35, 0.35), (-0.6, 0.6)],
    1: [(-0.35, 0.3), (-0.5, 0.5)],
    2: [(-0.2, 0.12), (-0.6, 0.6)],
    3: [(-1, 1), (-1, 1)]
}

directions = ["front", "behind", "right", "left", "top"]