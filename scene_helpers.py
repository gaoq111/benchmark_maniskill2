import os
import shutil
import random
import numpy as np

def create_directories(data_dir, empty=False):
    final_dir = f"{data_dir}/final"
    os.makedirs(final_dir, exist_ok=True)
    if empty:
        shutil.rmtree(final_dir)
        os.makedirs(final_dir, exist_ok=True)
    return final_dir

def create_config(obj_type, color, size, position_range, static=True, name=None):
    config = {
        'size': size,
        'color': color,
        'obj_type': obj_type,
        'ranges': position_range,
        'static': static
    }
    if name:
        config['name'] = name
    return config

def finalize_quadrant(quadrant, scale=1, num_divisions=4):
    lim_x, lim_y = quadrant
    
    # Divide the quadrant into a grid
    x_step = (lim_x * scale) / num_divisions
    y_step = (lim_y * scale) / num_divisions
    
    # Choose a random cell in the grid
    x_index = random.randint(0, num_divisions - 1)
    y_index = random.randint(0, num_divisions - 1)
    
    # Calculate the range for the chosen cell
    x_range = [x_index * x_step, (x_index + 1) * x_step]
    y_range = [y_index * y_step, (y_index + 1) * y_step]
    
    x_range.sort()
    y_range.sort()
    interval_x = np.array(x_range)*scale
    interval_y = np.array(y_range)*scale
    return interval_x, interval_y

def get_random_position(background_quadrants, background, scale=0.6):
    quadrant = random.choice(background_quadrants[background])
    return finalize_quadrant(quadrant, scale=scale)