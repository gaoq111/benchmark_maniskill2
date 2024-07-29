import numpy as np
from utils import *
from agent_env_utils import *
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



'''
Helper functions for Spatial Reasoning Scenarios in [Perception]
'''


'''
=======================================================================================================
Spatial Occupancy
=======================================================================================================
'''
def generate_volume_comparison(env, obj_infos, save_info):
    outputs = []
    configs = []
    episode_id = save_info['episode_id']
    final_dir = save_info['output_dir']
    r = save_info['round']

    obj_names = obj_infos['obj_name']
    size_categories = obj_infos['size_categories']
    sizes = [round(np.random.uniform(num*0.02, num*0.02+0.01, 1).tolist()[0], 5) for num in size_categories ]
    colors = obj_infos['color']
    color_ids = [color_maps[color] for color in colors]
    base_size =  0.08

    ### Create configurations (Obj_id example: 026_sponge)
    bounds = []
    for i, obj_name in enumerate(obj_names):
        obj_type = obj_name.split("-")[0]
        obj_id = obj_name.split("-")[-1]
        if(obj_type == "custom"):
            actual_size = sizes[i]*1.1/2
        else:
            actual_size = sizes[i]

        if(obj_id == "bound"):
            bounds.append(i)

        configs.append({'obj_type': obj_type, 'name': obj_id, 'size': actual_size, 'color': color_ids[i], 
                        'ranges': [(-0.1, 0.25), (-0.3, -0.2)], 'static': False})

    ### Extra configs for bounding objects
    extra_configs = []
    for bound_id in bounds:
        for i in range(7):
            extra_configs.append(configs[bound_id].copy())
    # extra_configs = [configs[0].copy() for _ in range(8)]
    # configs += extra_configs

    env.register_configures(collate_infos(configs))
    if(len(extra_configs) != 0):
        env.register_configures(collate_infos(extra_configs), replace=False)
    _ = env.reset(options={"reconfigure": True})

    env.initalize_objects(regular=True)
    
    main_placements = []
    for i, obj_config in enumerate(configs):
        if(i+1 >= len(configs)):
            break
        main_placements.append((i+1, i, "right")) #, (1, 0, "top"), (1, 0, "front")]

    
    distances = [np.random.uniform(3*base_size, 3.5*base_size, size=1)[0] * np.sqrt(2)]

    for direction_config in main_placements:
        if(direction_config[2] == "top" and obj_type == "custom"):
            env.place_cubes_in_direction(direction_config, distances=distances+2*base_size,)
        else:
            env.place_cubes_in_direction(direction_config, distances=distances)
        
    ### Prepare bounded areas
    for k,bound_id in enumerate(bounds):
        ### Border
        id_offset = k*7 + len(configs)
        bound_direction_configs = [(0+id_offset, bound_id, "left"), (bound_id, bound_id, "right"), 
                                   (1+id_offset, 0+id_offset, "behind"), (2+id_offset, 1+id_offset, "right"), (3+id_offset, 2+id_offset, "right"),
                                    (4+id_offset, 0+id_offset, "front"), (5+id_offset, 4+id_offset, "right"), (6+id_offset, 5+id_offset, "right"),]
        env.place_cubes_in_direction(bound_direction_configs, distances=[2*configs[bound_id]['size'] for _ in bound_direction_configs])
        # ### Extra Border
        # bound_direction_configs = [(3, 3, "front"), (4, 4, "front"), 
        #                             (7, 7, "behind"), (8, 8, "behind"), ]
        # env.place_cubes_in_direction(bound_direction_configs, 
        #                                 distances=[size/5*(3-(k%(len(bound_direction_configs)//2)+1)) for k in range(len(bound_direction_configs))])
        bound_ids = [bound_id]
        
    save_dir_initial = f"{final_dir}/volume{episode_id}_round{r}.png"
    collect_and_save(env, save_dir_initial, steps=5, mode="all")

    # poses_initial = env.get_important_obj_poses()
    # save_dir_initial = f"{final_dir}/color{color_idx}_{r}_{color_1}_{direction}_{color_2}_ini_10.png"
    # collect_and_save(env, save_dir_initial, steps=10, mode="all")

    outputs.append({
        "source": save_dir_initial,
        "obj_names": obj_names,
        'sizes': sizes,
        "colors": colors,
        "round": r,
        #"poses_final": [list(np.float64(item)) for item in poses_initial],
    })
    return outputs

def generate_volume_comparison(env, obj_infos, save_info):
    outputs = []
    configs = []
    episode_id = save_info['episode_id']
    final_dir = save_info['output_dir']
    r = save_info['round']

    obj_names = obj_infos['obj_name']
    size_categories = obj_infos['size_categories']
    sizes = [round(np.random.uniform(num*0.02, num*0.02+0.01, 1).tolist()[0], 5) for num in size_categories ]
    colors = obj_infos['color']
    color_ids = [color_maps[color] for color in colors]
    base_size =  0.08

    ### Create configurations (Obj_id example: 026_sponge)
    bounds = []
    for i, obj_name in enumerate(obj_names):
        obj_type = obj_name.split("-")[0]
        obj_id = obj_name.split("-")[-1]
        if(obj_type == "custom"):
            actual_size = sizes[i]*1.1/2
        else:
            actual_size = sizes[i]

        if(obj_id == "bound"):
            bounds.append(i)

        configs.append({'obj_type': obj_type, 'name': obj_id, 'size': actual_size, 'color': color_ids[i], 
                        'ranges': [(-0.1, 0.25), (-0.3, -0.2)], 'static': False})

    ### Extra configs for bounding objects
    extra_configs = []
    for bound_id in bounds:
        for i in range(7):
            extra_configs.append(configs[bound_id].copy())
    # extra_configs = [configs[0].copy() for _ in range(8)]
    # configs += extra_configs

    env.register_configures(collate_infos(configs))
    if(len(extra_configs) != 0):
        env.register_configures(collate_infos(extra_configs), replace=False)
    _ = env.reset(options={"reconfigure": True})

    env.initalize_objects(regular=True)
    
    main_placements = []
    for i, obj_config in enumerate(configs):
        if(i+1 >= len(configs)):
            break
        main_placements.append((i+1, i, "right")) #, (1, 0, "top"), (1, 0, "front")]

    
    distances = [np.random.uniform(3*base_size, 3.5*base_size, size=1)[0] * np.sqrt(2)]

    for direction_config in main_placements:
        if(direction_config[2] == "top" and obj_type == "custom"):
            env.place_cubes_in_direction(direction_config, distances=distances+2*base_size,)
        else:
            env.place_cubes_in_direction(direction_config, distances=distances)
        
    ### Prepare bounded areas
    for k,bound_id in enumerate(bounds):
        ### Border
        id_offset = k*7 + len(configs)
        bound_direction_configs = [(0+id_offset, bound_id, "left"), (bound_id, bound_id, "right"), 
                                   (1+id_offset, 0+id_offset, "behind"), (2+id_offset, 1+id_offset, "right"), (3+id_offset, 2+id_offset, "right"),
                                    (4+id_offset, 0+id_offset, "front"), (5+id_offset, 4+id_offset, "right"), (6+id_offset, 5+id_offset, "right"),]
        env.place_cubes_in_direction(bound_direction_configs, distances=[2*configs[bound_id]['size'] for _ in bound_direction_configs])
        # ### Extra Border
        # bound_direction_configs = [(3, 3, "front"), (4, 4, "front"), 
        #                             (7, 7, "behind"), (8, 8, "behind"), ]
        # env.place_cubes_in_direction(bound_direction_configs, 
        #                                 distances=[size/5*(3-(k%(len(bound_direction_configs)//2)+1)) for k in range(len(bound_direction_configs))])
        bound_ids = [bound_id]
        
    save_dir_initial = f"{final_dir}/volume{episode_id}_round{r}.png"
    collect_and_save(env, save_dir_initial, steps=5, mode="all")

    # poses_initial = env.get_important_obj_poses()
    # save_dir_initial = f"{final_dir}/color{color_idx}_{r}_{color_1}_{direction}_{color_2}_ini_10.png"
    # collect_and_save(env, save_dir_initial, steps=10, mode="all")

    outputs.append({
        "source": save_dir_initial,
        "obj_names": obj_names,
        'sizes': sizes,
        "colors": colors,
        "round": r,
        #"poses_final": [list(np.float64(item)) for item in poses_initial],
    })
    return outputs