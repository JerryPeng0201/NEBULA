from typing import Any, Dict, Union

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import random
import sapien
import torch
import torch.random
from transforms3d.euler import euler2quat

from nebula.core.embodiment.robots import Fetch, Panda
from nebula.core.simulation.engine import BaseEnv
from nebula.core.simulation.utils import randomization
from nebula.core.sensors.camera import CameraConfig
from nebula.utils import common, sapien_utils
from nebula.utils.building import actors
from nebula.utils.registration import register_env
from nebula.utils.scene_builder.table import TableSceneBuilder
from nebula.utils.structs import Pose
from nebula.utils.structs.types import Array, GPUMemoryConfig, SimConfig


@register_env("Language-Straight-Easy", max_episode_steps=50)
class LanguageStraightEasyEnv(BaseEnv):
    """
    **Task Description:**
    Grasp the red cube among three objects (blue cube, red cube, and green cube) based on 
    language instructions. This is a language grounding task that tests the robot's ability 
    to understand natural language commands with varying verb choices.

    **Language Commands:**
    - Random selection from: "{Grab/Pick/Select} red block"
    - Tests understanding of different verbs for the same action
    - Uses "block" as a synonym for "cube"

    **Randomizations:**
    - The objects are placed at fixed positions: blue cube at x=-0.08, y=-0.05,
    red cube at x=-0.08, y=0.05, and green cube at x=-0.08, y=0.15.
    - Language command verb is randomly selected each episode

    **Success Conditions:**
    - The red cube is grasped by the robot based on the language command
    """

    SUPPORTED_ROBOTS = ["panda", "fetch"]

    # Specify some supported robot types
    agent: Union[Panda, Fetch]

    # Object dimensions
    blue_cube_half_size = 0.02      # half size of the blue cube
    red_cube_half_size = 0.02       # half size of the red cube
    green_cube_half_size = 0.02     # half size of the green cube

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        
        # Language templates with different verbs
        self.command_templates = [
            "Grab red block",
            "Pick red block",
            "Select red block",
        ]
        
        # Initialize task_instruction to empty string (will be set during reset)
        self.task_instruction = ""
        
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18
            )
        )

    @property
    def _default_sensor_configs(self):
        # Base camera configuration
        base_pose = sapien_utils.look_at(eye=[0.3, 0, 0.2], target=[-0.1, 0, 0])
        base_camera = CameraConfig(
            "base_camera",
            pose=base_pose,
            width=512,
            height=512,
            fov=np.pi / 2,
            near=0.01,
            far=100,
        )
        
        # Hand camera configuration
        hand_camera = CameraConfig(
            uid="hand_camera",
            pose=sapien.Pose(p=[0, 0, -0.05], q=[0, 0.7071, 0, 0.7071]),
            width=512,
            height=512,
            fov=np.pi / 2,
            near=0.01,
            far=100,
            mount=self.agent.robot.links_map["panda_hand_tcp"],
        )

        # Back right camera
        back_right_pose = sapien_utils.look_at(eye=[-0.3, -0.3, 0.3], target=[0, 0, 0.1])
        back_right_camera = CameraConfig(
            uid="back_right_camera",
            pose=back_right_pose,
            width=512,
            height=512,
            fov=np.pi / 3,
            near=0.01,
            far=2.0,
        )

        # Back left camera
        back_left_pose = sapien_utils.look_at(eye=[-0.3, 0.3, 0.3], target=[0, 0, 0.1])
        back_left_camera = CameraConfig(
            uid="back_left_camera",
            pose=back_left_pose,
            width=512,
            height=512,
            fov=np.pi / 3,
            near=0.01,
            far=2.0,
        )

        # Front right camera
        front_right_pose = sapien_utils.look_at(eye=[0.5, -0.5, 0.3], target=[0, 0, 0.1])
        front_right_camera = CameraConfig(
            uid="front_right_camera",
            pose=front_right_pose,
            width=512,
            height=512,
            fov=np.pi / 3,
            near=0.01,
            far=2.0,
        )

        # Front left camera
        front_left_pose = sapien_utils.look_at(eye=[0.5, 0.5, 0.3], target=[0, 0, 0.1])
        front_left_camera = CameraConfig(
            uid="front_left_camera",
            pose=front_left_pose,
            width=512,
            height=512,
            fov=np.pi / 3,
            near=0.01,
            far=2.0,
        )
        
        return [base_camera, hand_camera, back_right_camera, back_left_camera, front_right_camera, front_left_camera]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, -0.2, 0.2], [0.0, 0.0, 0.2])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
        )

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        # load the table
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # load the blue cube, red cube, and green cube
        self.blue_cube = actors.build_cube(
            self.scene,
            half_size=self.blue_cube_half_size,
            color=np.array([12, 42, 160, 255]) / 255,
            name="blue_cube",
            body_type="dynamic",
            initial_pose=Pose.create_from_pq(p=[0, 0, self.blue_cube_half_size], q=[1, 0, 0, 0])
        )
        
        self.red_cube = actors.build_cube(
            self.scene,
            half_size=self.red_cube_half_size,
            color=np.array([255, 0, 0, 255]) / 255,
            name="red_cube",
            body_type="dynamic",
            initial_pose=Pose.create_from_pq(p=[0, 0, self.red_cube_half_size], q=[1, 0, 0, 0])
        )
        
        self.green_cube = actors.build_cube(
            self.scene,
            half_size=self.green_cube_half_size,
            color=np.array([0, 255, 0, 255]) / 255,
            name="green_cube",
            body_type="dynamic",
            initial_pose=Pose.create_from_pq(p=[0, 0, self.green_cube_half_size], q=[1, 0, 0, 0])
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            # init the table scene
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # Randomly select a language command from templates
            self.task_instruction = random.choice(self.command_templates)
            print(f"Language Command: '{self.task_instruction}'")

            # init the objects at fixed positions
            xyz = torch.zeros((b, 3))
            q = [1, 0, 0, 0]
            
            # Blue cube position (fixed)
            xyz[..., 0] = -0.08  # fixed x position
            xyz[..., 1] = -0.05  # fixed y position  
            xyz[..., 2] = self.blue_cube_half_size  # on the table (cube half size for bottom)
            blue_cube_pose = Pose.create_from_pq(p=xyz.clone(), q=q)
            self.blue_cube.set_pose(blue_cube_pose)

            # Red cube position (fixed, with guaranteed distance)
            xyz[..., 0] = -0.08  # same x as blue cube
            xyz[..., 1] = 0.05   # fixed y position, 0.1 units away from blue cube
            xyz[..., 2] = self.red_cube_half_size  # on the table (cube half size for bottom)
            red_cube_pose = Pose.create_from_pq(p=xyz.clone(), q=q)
            self.red_cube.set_pose(red_cube_pose)

            # Green cube position (fixed)
            xyz[..., 0] = -0.08  # same x as blue and red cube
            xyz[..., 1] = 0.15   # fixed y position, 0.1 units away from red cube
            xyz[..., 2] = self.green_cube_half_size  # on the table (cube half size for bottom)
            green_cube_pose = Pose.create_from_pq(p=xyz, q=q)
            self.green_cube.set_pose(green_cube_pose)

    def evaluate(self):
        # Check if objects are grasped
        is_blue_cube_grasped = self.agent.is_grasping(self.blue_cube)
        is_red_cube_grasped = self.agent.is_grasping(self.red_cube)
        is_green_cube_grasped = self.agent.is_grasping(self.green_cube)
        
        # Success: grasp the red cube - language task requires grasping the correct object
        success = is_red_cube_grasped
        
        return {
            "is_blue_cube_grasped": is_blue_cube_grasped,
            "is_red_cube_grasped": is_red_cube_grasped,
            "is_green_cube_grasped": is_green_cube_grasped,
            "success": success,
            "task_instruction": self.task_instruction,  # Store language command in info
        }

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            is_blue_cube_grasped=info["is_blue_cube_grasped"],
            is_red_cube_grasped=info["is_red_cube_grasped"],
            is_green_cube_grasped=info["is_green_cube_grasped"],
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )
        if "state" in self.obs_mode:
            obs.update(
                blue_cube_pose=self.blue_cube.pose.raw_pose,
                red_cube_pose=self.red_cube.pose.raw_pose,
                green_cube_pose=self.green_cube.pose.raw_pose,
                tcp_to_blue_cube_pos=self.blue_cube.pose.p - self.agent.tcp.pose.p,
                tcp_to_red_cube_pos=self.red_cube.pose.p - self.agent.tcp.pose.p,
                tcp_to_green_cube_pos=self.green_cube.pose.p - self.agent.tcp.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # reaching reward - reward for getting close to the RED CUBE (the target)
        tcp_pose = self.agent.tcp.pose.p
        red_cube_pos = self.red_cube.pose.p
        
        obj_to_tcp_dist_red = torch.linalg.norm(tcp_pose - red_cube_pos, axis=1)
        reward = 2 * (1 - torch.tanh(5 * obj_to_tcp_dist_red))

        # grasp reward - reward for grasping the correct (red cube) object
        is_red_grasped = info["is_red_cube_grasped"]
        reward[is_red_grasped] = 5.0

        # success reward - successfully grasped the red cube
        reward[info["success"]] = 10.0
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
        # this should be equal to compute_dense_reward / max possible reward
        max_reward = 10.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward
    
    def get_task_instruction(self):
        """Return the current language instruction for this episode"""
        return self.task_instruction