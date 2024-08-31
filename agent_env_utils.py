import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
import sys
sys.path.append('..')
from PIL import Image
from tqdm import trange
#from data_utils import get_nth_episode_info
from IPython.display import clear_output
from sapien.core import Pose
from mani_skill2.envs.pick_and_place.base_env import StationaryManipulationEnv
from mani_skill2.utils.registration import register_env
import gymnasium as gym
import sapien.core as sapien
from pathlib import Path
from typing import Dict, List
from mani_skill2 import format_path
from mani_skill2.utils.io_utils import load_json
import numpy as np
from sapien.core import Pose
import matplotlib.pyplot as plt
from mani_skill2.agents.base_agent import BaseAgent
import os
import math
from collections import defaultdict



# Register ManiSkill2 environments in gym
import mani_skill2.envs
from mani_skill2.utils.sapien_utils import look_at


from mani_skill2.utils.registration import register_env

from mani_skill2.envs.pick_and_place.pick_cube import PickCubeEnv
from mani_skill2.envs.pick_and_place.stack_cube import StackCubeEnv
from mani_skill2.envs.pick_and_place.pick_single import PickSingleYCBEnv
from mani_skill2.envs.pick_and_place.pick_clutter import PickClutterYCBEnv
from mani_skill2.envs.assembly.peg_insertion_side import PegInsertionSideEnv
from mani_skill2.envs.assembly.plug_charger import PlugChargerEnv
from mani_skill2.envs.assembly.assembling_kits import AssemblingKitsEnv
from mani_skill2.envs.misc.turn_faucet import TurnFaucetEnv
from mani_skill2.envs.ms1.push_chair import PushChairEnv
from mani_skill2.envs.ms1.open_cabinet_door_drawer import OpenCabinetDoorEnv, OpenCabinetDrawerEnv
from mani_skill2.envs.ms1.move_bucket import MoveBucketEnv

from mani_skill2.utils.sapien_utils import hex2rgba
from mani_skill2.sensors.camera import CameraConfig

from transforms3d.euler import euler2quat, quat2euler
from mani_skill2.envs.pick_and_place.pick_cube import PickCubeEnv
import itertools
from mani_skill2.utils.sapien_utils import vectorize_pose
from collections import OrderedDict

from gymnasium import spaces
from mani_skill2.agents.base_controller import BaseController

ASSET_DIR = "/root/maniskill2/data"


def save_obs(obs, save_dir, mode):
    if mode == "all":
        camera_keys = obs['image'].keys()
    elif mode == "front":
        camera_keys = ["front"]
    else:
        camera_keys = mode

    for camera_key in camera_keys:
        try:
            image_data = obs['image'][camera_key]['rgb']
            Image.fromarray(image_data.astype(np.uint8)).save(save_dir.split(".png")[0] + f"_{camera_key}.png")
        except KeyError as e:
            print(f"KeyError: {e} - Available keys: {obs['image'].keys()}")
            print(f"Full observation structure: {obs}")

def collect_and_save(env, save_dir, steps, mode):
    for i in range(steps):
        obs, _, _, _, _ = env.step(np.zeros(len(env.action_space.sample())))
    save_obs(obs, save_dir, mode=mode)


def collate_infos(info_pairs):
    infos = defaultdict(list)
    for pair in info_pairs:
        for key, value in pair.items():
            infos[key].append(value)
    return dict(infos)


def is_overlapping(pos1, pos2, min_dist):
    # Calculate the Euclidean distance between two positions
    dist = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    return dist < min_dist

def generate_non_overlapping_position(existing_positions, min_dist, ranges=[(-0.3, 0.3), (-0.3, 0.3)]):
    t = 0
    while True and t < 5:
        # Generate a random position
        new_position = []
        for i in range(len(ranges)):
            new_position.append(np.random.uniform(low=min(ranges[i]), high=max(ranges[i]), size=1)[0])
        # Check for overlap with existing positions
        overlap = any(is_overlapping(new_position, pos, min_dist) for pos in existing_positions)
        if not overlap:
            return new_position
        t += 1

    return new_position
        
def check_move(pos1, pos2, threshold=0.3):
    return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2 + (pos1[2] - pos2[2])**2) >= threshold

def create_steps(start, end, step):
    result = []
    current = start
    while current <= end:
        result.append(step)
        current += step
    if current - step != end:
        result.append(end+step-current)
    return result

class DummyAgent(BaseAgent):

    robot: sapien.Articulation
    controllers: Dict[str, BaseController]

    def __init__(
        self,
    ):
        self.scene = None
        self._control_freq = None

        self.config = None

        # URDF
        self.urdf_path = None
        self.fix_root_link = None

    
    @property
    def action_space(self):
        # if self._control_mode is None:
        #     return spaces.Dict(
        #         {
        #             uid: controller.action_space
        #             for uid, controller in self.controllers.items()
        #         }
        #     )
        # else:
        #     return self.controller.action_space
        return spaces.Dict()

    def reset(self, init_qpos=None):
        pass

    def set_action(self, action):
        #if np.isnan(action).any(): raise ValueError("Action cannot be NaN. Environment received:", action)
        pass

    def before_simulation_step(self):
        pass

    # -------------------------------------------------------------------------- #
    # Observations
    # -------------------------------------------------------------------------- #
    def get_proprioception(self):
        obs = OrderedDict(qpos=self.robot.get_qpos(), qvel=self.robot.get_qvel())
        controller_state = self.controller.get_state()
        if len(controller_state) > 0:
            obs.update(controller=controller_state)
        return obs

    def get_state(self) -> Dict:
        """Get current state for MPC, including robot state and controller state"""
        state = OrderedDict()

        return state

    def set_state(self, state: Dict, ignore_controller=False):
        # robot state
        pass



@register_env("CustomEnv-v0", max_episode_steps=200, override=True)
class CustomEnv(PickCubeEnv):
    def __init__(self, *args, obj_init_rot_z=True, **kwargs):
        self.obj_builds = {
            "cube": self.build_regular,
            "sphere": self.build_regular,
            "custom": self.build_asset,
        }
        self.sphere_radius = 0.1
        self.objects = []
        self.infos = {'size': [0.1, 0.1], 'color': [(1, 0, 0), (0, 1, 0)], 'obj_type': ["sphere", "sphere"], 'static': [True, True]}
        self.builder = None

        self.direction_placement_map = {
            "behind": [-1, 0, 0],
            "front": [1, 0, 0],
            "left": [0, -1, 0],
            "right": [0, 1, 0],
            "top": [0, 0, 1/np.sqrt(2)],
        }
        self.table_centers = []
        super().__init__(*args, **kwargs)

    def _load_actors(self, actors=False):            
        #self._add_ground(render=self.bg_name is None)
        
        if(self.builder is None):
            self.builder = self._scene.create_actor_builder()
        
        builder = self._scene.create_actor_builder()
        path = f"{ASSET_DIR}/hab2_bench_assets/stages/Baked_sc1_staging_00.glb"
        pose = sapien.Pose(q=[0.707, 0.707, 0, 0])  # y-axis up for Habitat scenes
        # NOTE: use nonconvex collision for static scene
        builder.add_nonconvex_collision_from_file(path, pose)
        builder.add_visual_from_file(path, pose)
        self.arena = builder.build_static()
        # Add offset so that the workspace is on the table
        offset = np.array([-2.0616, -3.1837, 0.66467 + 0.095])
        self.arena.set_pose(sapien.Pose(-offset))
    

        #if(actors):
        # self.obj = self._build_cube(self.cube_half_size*self.size_1, color=self.color_1, static=False)
        # self.obj2 = self._build_cube(self.cube_half_size*self.size_2, color=self.color_2, static=False)

        self.generate_objects(self.infos)
        self.obj = self.objects[0]

        self.goal_site = self._build_sphere_site(self.goal_thresh)
    
    def reverse_collate(self, infos):
        output = []
        for key in infos:
            for i, info in enumerate(infos[key]):
                if i >= len(output):
                    output.append({})
                output[i][key] = info
        return output

    def generate_objects(self, infos=None):
        if(infos == None):
            infos = self.infos
        infos = self.reverse_collate(infos)
        self.objects = [self.obj_builds[info['obj_type']](info) for info in infos]

    def build_regular(self, config):
        size = config['size']
        color=config['color']
        static=config['static'] if 'static' in config else False
        if config['obj_type'] == "cube":
            return self._build_cube(size, color, static=static)
        elif config['obj_type'] == "sphere":
            return self._build_sphere(size, color, static=static)

    def build_asset(self, config):
        scale = config["size"]
        name = config["name"]
        """
            Configs: dict[
                "name": "cube" / "sphere" / "002_master_chef_can"
                "scale": float or none
                "color": tuple or none
            ]
        """
        if(self.builder is None):
            self.builder = self._scene.create_actor_builder()
        builder = self._scene.create_actor_builder()
        collision_file = f"{ASSET_DIR}/mani_skill2_ycb/models/{name}/collision.obj"
        scale = np.array([scale/ 0.01887479572529618/2 for _ in range(3)])
        builder.add_multiple_collisions_from_file(
            filename=collision_file, scale=scale/1.2, density=1000
        )
        visual_file = f"{ASSET_DIR}/mani_skill2_ycb/models/{name}/textured.obj" 
        builder.add_visual_from_file(filename=visual_file, scale=scale)
        # filepath = f"models/{name}/mesh.obj"
        # scale = [scale] #*= self.cube_half_size / 0.01887479572529618 / 3
        # builder.add_multiple_collisions_from_file(
        #     filename=filepath, scale=scale, density=1000
        # )
        # builder.add_visual_from_file(filename=filepath, scale=scale)

        if "static" in config and config["static"]:
            obj = builder.build_static(name=name)
        else:
            obj = builder.build(name=name)
            obj.lock_motion(False, False, False, True, True, True)

        return obj
    
    def _load_agent(self):
        self.agent = DummyAgent()
        self.tcp  = None #sapien.Link()

    def _initialize_actors(self, actors=False):
        #if(actors):
        super()._initialize_actors()
        
    
    def _configure_agent(self):
        self._agent_cfg = None
        pass

    def _initialize_agent(self):
        pass

    def _get_obs_agent(self):
        return OrderedDict()
    
    def _get_obs_extra(self) -> OrderedDict:
        obs = OrderedDict()
        if self._obs_mode in ["state", "state_dict"]:
            obs.update(
                obj_pose=vectorize_pose(self.obj.pose),
                tcp_to_obj_pos=self.obj.pose.p - self.tcp.pose.p,
            )
        return OrderedDict() #obs
    
    def customize_camera(self, p, reference_point, name):
        cam_cfg = super()._register_render_cameras()
        cam_cfg.p = p
        pose3 = look_at(cam_cfg.p, reference_point)
        cam_cfg.q = pose3.q
        cam_cfg.fov = 1.5
        cam_cfg.uid = name

        return cam_cfg

    def _register_cameras(self, background_num=1):
        pose = look_at([-1, 0, 1], [0, 0, 0])
        cfg1 = CameraConfig(
            "base_camera", pose.p, pose.q, 512, 512, np.pi/3, 0.01, 10
        )
        cam_cfg = super()._register_render_cameras()
        ### View that can see all the table
        #table_center = [0.70000001-0.2, 0.40000001-0.4, 0.50500001-0.1]
        #cam_cfg.p = [1.00000001e+00, 5.96046446e-09, 6.05000012e-01]

        ### Only table
        table_center = [0.70000001-0.7, 0.40000001-0.39, 0]
        table_center2 = [4.4, -2.88,  0.2]
        table_center3 = [5.95,  1.815, -0.15]
        table_center4 = [4.1, 1.41, -0.72]
        fsp_points = {
            "front": [0.8, 5.96046446e-09+0.03, 6.05000012e-01-0.3],
            "side": [1.00000001e+00-1, 5.96046446e-09+1.1, 6.05000012e-01-0.18],
            "top": [1.00000001e+00-0.7, 5.96046446e-09+0.01, 6.05000012e-01+0.25],
        }
        fsp_points2 = {
            "front": [5.2, -2.9,  0.5],
            "side": [4.4, -1.8,  0.5],
            "top": [5.2, -2.9,  1],
        }
        fsp_points3 = {
            "side": [5.3, 1.81, 0.1],
            "front": [5.9, 0.5, 0.1],
            "top": [5.5, 1.81, 0.6],
        }
        fsp_points4 = {
            "front": [4, 0, -0.3],
            "side": [5.2, 1.5, -0.25],
            "top": [4, 1.42, 0.805],
        }
        table_sets = [(table_center, fsp_points), (table_center2, fsp_points2), (table_center3, fsp_points3), (table_center4, fsp_points4)]

        cam_cfg.p = [1.00000001e+00, 5.96046446e-09+0.03, 6.05000012e-01-0.08]
        
        #cam_cfg.p = cam_cfg.p + [0.5+0.1, 0.5-0.9, -0.095-0.1]
        pose2 = look_at(cam_cfg.p, table_center)
        cam_cfg.q = pose2.q
    
        #cam_cfg.p = pose.p
        cam_cfg.fov = 1.5
        cam_cfg.uid = "front_prev"
        camera_configs = [cfg1, cam_cfg]

        self.table_centers = []
        for k, (tab_center, fsp_set) in enumerate(table_sets):
        

            #cam_cfg_front = self.customize_camera([1.00000001e+00, 5.96046446e-09+0.03, 6.05000012e-01-0.3], reference_point=table_center, name="front")
            #cam_cfg_side = self.customize_camera([1.00000001e+00-1, 5.96046446e-09+1.1, 6.05000012e-01-0.18], reference_point=table_center, name="side")
            #cam_cfg_top = self.customize_camera([1.00000001e+00-0.35, 5.96046446e-09+0.01, 6.05000012e-01+0.25], reference_point=table_center, name="top")
            for key in fsp_set:
                cam_cfg = self.customize_camera(fsp_set[key], reference_point=tab_center, name=f"{key}{k}")
                
                camera_configs.append(cam_cfg)
            self.table_centers.append(tab_center)

        return camera_configs

    # def reconfig_camera(self, background=0):
    #     self._register_cameras(background)
    #     self._configure_cameras()

    def evaluate(self, **kwargs):
        is_obj_placed = False #self.check_obj_placed()
        is_robot_static = False #self.check_robot_static()
        return dict(
            is_obj_placed=is_obj_placed,
            is_robot_static=is_robot_static,
            success=is_obj_placed and is_robot_static,
        )
    def compute_normalized_dense_reward(self, **kwargs):
        return 0
    
    def register_configures(self, infos, replace=True):
        if(not replace):
            for key in self.infos:
                self.infos[key] = self.infos[key] + infos[key]
        else:
            self.infos = infos        
        #self.reconfigure()

    def initialize_objects(self, regular=False, existing_positions = [], background=0):
        # min_dist = 2 * 4 * np.sqrt(2) * env.cube_half_size[-1]

        sizes = self.infos['size']
        ranges = self.infos['ranges']

        min_dist = 2*max(sizes) * np.sqrt(2)
        table_center = self.table_centers[background]
        self.background = background

        for i in range(len(sizes)):
            position = generate_non_overlapping_position(existing_positions, min_dist, ranges=ranges[i])
            if(regular):
                pos_angle = 0
            else:
                pos_angle = np.random.uniform(-np.pi*2, np.pi*2)
            self.objects[i].set_pose(Pose([position[0]+table_center[0], position[1]+table_center[1], sizes[i]*2+table_center[2]], euler2quat(0, 0, pos_angle)))
            existing_positions.append(position)

    def spawn_next(self, place_info, distance=None):
        sizes = self.infos['size']
        move_obj, ref_obj, direction = place_info
        if(distance is None):
            distance = (sizes[move_obj] + sizes[ref_obj]) * np.sqrt(2)

        pose_obj_ref = self.objects[ref_obj].get_pose()


        if(len(direction.split("_")) == 1):
            ### simple direction here
            new_position = [dp*distance for dp in self.direction_placement_map[direction]]
        else:
            ### complex direction here
            directions = direction.split("_")
            new_position = sum([np.array(self.direction_placement_map[direction])*distance/len(directions) for direction in directions]).tolist()

        new_position = [pair[0]+pair[1] for pair in zip(pose_obj_ref.p, new_position)]
        
        #change the height of the object
        #new_position[2] = sizes[move_obj]*2 + self.table_centers[self.background][2]
        
                
        self.objects[move_obj].set_pose(Pose(new_position, self.objects[move_obj].get_pose().q))

    def place_cubes_in_direction(self, place_infos, distances=[]):
        sizes = self.infos['size']
        if(type(place_infos) is not list):
            place_infos = [place_infos]
        for i, info in enumerate(place_infos):
            move_obj, ref_obj, direction = info
            if(len(distances) > i):
                distance = distances[i]
            else:
                distance = (sizes[move_obj] + sizes[ref_obj]) * np.sqrt(2)*1.2

            pose_obj_ref = self.objects[ref_obj].get_pose()


            if(len(direction.split("_")) == 1):
                ### simple direction here
                new_position = [dp*distance for dp in self.direction_placement_map[direction]]
            else:
                ### complex direction here
                directions = direction.split("_")
                new_position = sum([np.array(self.direction_placement_map[direction])*distance/len(directions) for direction in directions]).tolist()

            new_position = [pair[0]+pair[1] for pair in zip(pose_obj_ref.p, new_position)]
                
            self.objects[move_obj].set_pose(Pose(new_position, self.objects[move_obj].get_pose().q))
        
    def place_obj_with_displacement(self, place_infos, displacement, distances=[]):
        sizes = self.infos['size']
        if(type(place_infos) is not list):
            place_infos = [place_infos]
        for i, info in enumerate(place_infos):
            move_obj, ref_obj, direction = info
            move_obj, ref_obj, direction = info
            if(len(distances) > i):
                distance = distances[i]
            else:
                distance = (sizes[move_obj] + sizes[ref_obj]) * np.sqrt(2)
            pose_obj_ref = self.objects[ref_obj].get_pose()
            displacement = [change*distance for change in displacement]

            new_position = [pair[0]+pair[1] for pair in zip(pose_obj_ref.p, displacement)]
                
            self.objects[move_obj].set_pose(Pose(new_position, self.objects[move_obj].get_pose().q))

    def get_circle_displacements(self, circle_type, scale):
        # Parameters
        r = 1 * scale
        delta_t = 0.2

        if(circle_type == "circle_lcc"):
            ### left circle; counter clockwise
            t_values = np.arange(0, 2*np.pi, delta_t)
        elif(circle_type == "circle_lc"):
            ### left circle; clockwise
            t_values = np.arange(0, -2*np.pi, -delta_t)
        elif(circle_type == "circle_rc"):
            ### right circle; clockwise
            t_values = np.arange(np.pi, -np.pi, -delta_t)
        elif(circle_type == "circle_rcc"):
            ### right circle; counter clockwise
            t_values = np.arange(-np.pi, np.pi, delta_t)

        # Positions
        x_values = r * np.cos(t_values)
        y_values = r * np.sin(t_values)

        # Displacements
        displacements = []
        for i in range(len(t_values) - 1):
            delta_x = x_values[i + 1] - x_values[i]
            delta_y = y_values[i + 1] - y_values[i]
            delta_z = 0
            displacements.append((delta_x, delta_y, delta_z))

        return displacements
        

    def move_in_directions(self, instructions, distances, steps=None, scale=1, camera_view="all", background=0):
        frames = {}
        views = (
        [f"side{background}", f"front{background}", f"top{background}"]
        if background == 3 else
        [f"front{background}", f"side{background}", f"top{background}"])
        if(camera_view == "all"):
            for view in views:
                frames[view] = []
        else:
            frames[camera_view] = []

        if(type(instructions[0]) is not list and type(instructions[0]) is not tuple):
            instructions = [instructions]
        if(type(distances) is not list):
            distances = [distances]
        distances = [dist*scale for dist in distances]
        #print("Initial position :")
        #print(self.obj.get_pose())
        #obs, _, _, _, _ = self.step(np.zeros(len(self.action_space.sample())))
        #frames.append(obs['image']['base_camera']['rgb'])

        for i,direction_info in enumerate(instructions):
            if(steps is None):
                step_size = distances[i]/10
            else:
                step_size = distances[i]/steps
            if("circle" in direction_info[-1]):
                step_sizes = self.get_circle_displacements(circle_type=direction_info[-1], scale=scale)
            else:
                step_sizes = create_steps(0, distances[i], step_size)
            # Move distance = step_size one at a time
            for i, step in enumerate(step_sizes):
                if("circle" in direction_info[-1]):
                    self.place_obj_with_displacement(direction_info, step, distances=[] ) #[distances[i]])
                else:  
                    self.place_cubes_in_direction(direction_info, distances=[step])
                obs, _, _, _, _ = self.step(np.zeros(len(self.action_space.sample())))
                if(camera_view == "all"):
                    for view in views:
                        frames[view].append(obs['image'][view]['Color'])
                else:
                    frames[camera_view].append(obs['image'][camera_view]['Color'])

        #print("Final position :")
        #print(self.obj.get_pose())

            # Save frames as GIF
            #imageio.mimsave('move_one_direction.gif', frames, duration=0.03)  # Set duration between frames (in seconds)
        if(camera_view != "all"):
            return frames[camera_view]
        return frames

    def get_important_obj_poses(self, mode="normal"):
        if(mode == "all"):
            return [obj.get_pose() for obj in self.objects]
        return [obj.get_pose().p for obj in self.objects]
    
    def set_poses(self, poses):
        for i,obj in enumerate(self.objects):
            obj.set_pose(poses[i])

    def _build_cube(
        self,
        half_size,
        color=(1, 0, 0),
        name="cube",
        static=False,
        render_material: sapien.RenderMaterial = None,
    ):
        half_size = [half_size for _ in range(3)]
        return super()._build_cube(half_size, color, name, static, render_material)

    def _build_sphere(self, radius, color=(0, 1, 0), name="sphere", static=False):
        """Build an actual sphere."""

        # if(self.builder is None):
        #     self.builder = self._scene.create_actor_builder()
        builder = self._scene.create_actor_builder()
        builder.add_sphere_collision(radius=radius)
        builder.add_sphere_visual(radius=radius, color=color)
        if static:
            return builder.build_static(name)
        else:
            return builder.build(name)
        
def initialize_obj_nooverlap_path(direction, configs, direction_length, env, distance_from_target = 0.2):
    configs_collate = collate_infos(configs)
    env.register_configures(configs_collate)
    _ = env.reset(options={"reconfigure": True})

    env.initialize_objects()
    obs, _, _, _, _ = env.step(np.zeros(len(env.action_space.sample())))

    initilization_poses = env.get_important_obj_poses(mode="all")
        
    diagonal_move = (direction_length + distance_from_target) / math.sqrt(2)
    ref_pos = initilization_poses[1]
    ref_info = np.array([ref_pos.p[0], ref_pos.p[1]])
    path_info = {
        "left": np.array([0, (direction_length + distance_from_target)]),
        "right": np.array([0, -(direction_length + distance_from_target)]),
        "front": np.array([-(direction_length + distance_from_target), 0]),
        "behind": np.array([(direction_length + distance_from_target), 0]),
        "left_front": np.array([- diagonal_move, diagonal_move]),
        "left_behind": np.array([diagonal_move, diagonal_move]),
        "right_front": np.array([- diagonal_move, - diagonal_move]),
        "right_behind": np.array([diagonal_move, - diagonal_move])
    }
    #get all position along path at that direction
    avoid_pos = [[ref_pos.p[0], ref_pos.p[1]]]
    for i in np.arange(0.01,1,0.01):
        avoid_pos += [list(ref_info + path_info[direction] * i)]
        
    env.initialize_objects(existing_positions = avoid_pos)
    obs, _, _, _, _ = env.step(np.zeros(len(env.action_space.sample())))
    initilization_poses = env.get_important_obj_poses(mode="all")
    initilization_poses[1] = ref_pos
        
    return env, initilization_poses, Pose(list(ref_info + path_info[direction]) + [ref_pos.p[2]], ref_pos.q)


