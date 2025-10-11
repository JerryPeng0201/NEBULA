from typing import Any, Dict, Union

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
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


@register_env("Perception-PickPeg-Hard", max_episode_steps=50)
class PerceptionPickPegHardEnv(BaseEnv):
    """
    **Task Description:**
    Grasp the blue-red-blue peg among two three-colored pegs with different patterns.
    This is a perception task that tests the robot's ability to identify the correct peg 
    based on its color pattern (blue-red-blue vs red-blue-red).

    **Randomizations:**
    - The pegs are placed at fixed positions: red-blue-red peg at y=-0.1 and 
    blue-red-blue peg at y=0.05, both at x=-0.08.

    **Success Conditions:**
    - The blue-red-blue peg is grasped by the robot. No placement is required for perception tasks.
    """

    SUPPORTED_ROBOTS = ["panda", "fetch"]

    # Specify some supported robot types
    agent: Union[Panda, Fetch]

    # set some commonly used values for pegs
    peg_half_length = 0.05  # half length of the peg
    peg_half_width = 0.02  # half width of the peg cross-section
    peg_height = 0.02  # height of the peg (2 * peg_half_width)

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
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

    def _build_three_color_peg(self, colors, name, initial_pos):
        """Build a three-block peg with specified colors [left, middle, right]"""
        builder = self.scene.create_actor_builder()
        
        # Set initial pose
        builder.initial_pose = sapien.Pose(p=initial_pos, q=[1, 0, 0, 0])
        
        # Each block is 1/3 of the total length
        block_length = self.peg_half_length * 2 / 3
        block_half_length = block_length / 2
        
        # Position centers for the three blocks
        # Left block center: -2/3 * peg_half_length
        # Middle block center: 0
        # Right block center: +2/3 * peg_half_length
        left_x = -2 * self.peg_half_length / 3
        middle_x = 0
        right_x = 2 * self.peg_half_length / 3
        
        # Add collision shapes for all three blocks
        builder.add_box_collision(
            sapien.Pose([left_x, 0, 0]),
            half_size=[block_half_length, self.peg_half_width, self.peg_half_width]
        )
        builder.add_box_collision(
            sapien.Pose([middle_x, 0, 0]),
            half_size=[block_half_length, self.peg_half_width, self.peg_half_width]
        )
        builder.add_box_collision(
            sapien.Pose([right_x, 0, 0]),
            half_size=[block_half_length, self.peg_half_width, self.peg_half_width]
        )
        
        # Add visual shapes with different colors
        # Left block
        mat_left = sapien.render.RenderMaterial()
        mat_left.set_base_color(colors[0])
        mat_left.roughness = 0.5
        mat_left.specular = 0.5
        builder.add_box_visual(
            sapien.Pose([left_x, 0, 0]),
            half_size=[block_half_length, self.peg_half_width, self.peg_half_width],
            material=mat_left,
        )
        
        # Middle block
        mat_middle = sapien.render.RenderMaterial()
        mat_middle.set_base_color(colors[1])
        mat_middle.roughness = 0.5
        mat_middle.specular = 0.5
        builder.add_box_visual(
            sapien.Pose([middle_x, 0, 0]),
            half_size=[block_half_length, self.peg_half_width, self.peg_half_width],
            material=mat_middle,
        )
        
        # Right block
        mat_right = sapien.render.RenderMaterial()
        mat_right.set_base_color(colors[2])
        mat_right.roughness = 0.5
        mat_right.specular = 0.5
        builder.add_box_visual(
            sapien.Pose([right_x, 0, 0]),
            half_size=[block_half_length, self.peg_half_width, self.peg_half_width],
            material=mat_right,
        )
        
        return builder.build_dynamic(name=name)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        # load the table
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # Define colors
        red_color = np.array([176, 14, 14, 255]) / 255
        blue_color = np.array([12, 42, 160, 255]) / 255

        # load the two three-block pegs with different color patterns
        # First peg: red-blue-red (blue in middle, red on sides)
        self.red_blue_red_peg = self._build_three_color_peg(
            colors=[red_color, blue_color, red_color],
            name="red_blue_red_peg",
            initial_pos=[-0.08, -0.1, self.peg_half_width]
        )
        
        # Second peg: blue-red-blue (red in middle, blue on sides) - TARGET
        self.blue_red_blue_peg = self._build_three_color_peg(
            colors=[blue_color, red_color, blue_color],
            name="blue_red_blue_peg",
            initial_pos=[-0.08, 0.05, self.peg_half_width]
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            # init the table scene
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            xyz = torch.zeros((b, 3))
            q = [1, 0, 0, 0]
            
            # Red-blue-red peg position (fixed)
            xyz[..., 0] = -0.08  # fixed x position
            xyz[..., 1] = -0.1  # fixed y position  
            xyz[..., 2] = self.peg_half_width  # on the table
            red_blue_red_peg_pose = Pose.create_from_pq(p=xyz.clone(), q=q)
            self.red_blue_red_peg.set_pose(red_blue_red_peg_pose)

            # Blue-red-blue peg position (fixed, with guaranteed distance)
            xyz[..., 0] = -0.08  # same x as first peg
            xyz[..., 1] = 0.05   # fixed y position, 0.15 units away from first peg
            xyz[..., 2] = self.peg_half_width  # on the table
            blue_red_blue_peg_pose = Pose.create_from_pq(p=xyz, q=q)
            self.blue_red_blue_peg.set_pose(blue_red_blue_peg_pose)

    def evaluate(self):
        # Check if pegs are grasped
        is_red_blue_red_peg_grasped = self.agent.is_grasping(self.red_blue_red_peg)
        is_blue_red_blue_peg_grasped = self.agent.is_grasping(self.blue_red_blue_peg)
        
        # Success: grasp the blue-red-blue peg - perception task only requires grasping the correct pattern
        success = is_blue_red_blue_peg_grasped
        
        return {
            "is_red_blue_red_peg_grasped": is_red_blue_red_peg_grasped,
            "is_blue_red_blue_peg_grasped": is_blue_red_blue_peg_grasped,
            "success": success,
        }

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            is_red_blue_red_peg_grasped=info["is_red_blue_red_peg_grasped"],
            is_blue_red_blue_peg_grasped=info["is_blue_red_blue_peg_grasped"],
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )
        if "state" in self.obs_mode:
            obs.update(
                red_blue_red_peg_pose=self.red_blue_red_peg.pose.raw_pose,
                blue_red_blue_peg_pose=self.blue_red_blue_peg.pose.raw_pose,
                tcp_to_red_blue_red_peg_pos=self.red_blue_red_peg.pose.p - self.agent.tcp.pose.p,
                tcp_to_blue_red_blue_peg_pos=self.blue_red_blue_peg.pose.p - self.agent.tcp.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # reaching reward - reward for getting close to the BLUE-RED-BLUE PEG (the target)
        tcp_pose = self.agent.tcp.pose.p
        blue_red_blue_peg_pos = self.blue_red_blue_peg.pose.p
        
        obj_to_tcp_dist_blue_red_blue = torch.linalg.norm(tcp_pose - blue_red_blue_peg_pos, axis=1)
        reward = 2 * (1 - torch.tanh(5 * obj_to_tcp_dist_blue_red_blue))

        # grasp reward - reward for grasping the correct (blue-red-blue) peg
        is_blue_red_blue_grasped = info["is_blue_red_blue_peg_grasped"]
        reward[is_blue_red_blue_grasped] = 5.0

        # success reward - successfully grasped the blue-red-blue peg
        reward[info["success"]] = 10.0
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
        # this should be equal to compute_dense_reward / max possible reward
        max_reward = 10.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward