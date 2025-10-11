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


@register_env("Perception-PickWhitePeg-Medium", max_episode_steps=50)
class PerceptionPickWhitePegMediumEnv(BaseEnv):
    """
    **Task Description:**
    Grasp the white peg (orange-white peg) among two two-colored pegs.
    This is a perception task that tests the robot's ability to identify and grasp the 
    peg with white color.

    **Randomizations:**
    - The pegs are placed at fixed positions: red-blue peg at y=-0.05 and 
    orange-white peg at y=0.05, both at x=-0.08.

    **Success Conditions:**
    - The orange-white peg is grasped by the robot. No placement is required for perception tasks.
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

    def _build_peg(self, color_1, color_2, name, initial_pos):
        """Build a two-color peg"""
        builder = self.scene.create_actor_builder()
        
        # Set initial pose
        builder.initial_pose = sapien.Pose(p=initial_pos, q=[1, 0, 0, 0])
        
        # Add collision shapes
        builder.add_box_collision(
            sapien.Pose([self.peg_half_length / 2, 0, 0]),
            half_size=[self.peg_half_length / 2, self.peg_half_width, self.peg_half_width]
        )
        builder.add_box_collision(
            sapien.Pose([-self.peg_half_length / 2, 0, 0]),
            half_size=[self.peg_half_length / 2, self.peg_half_width, self.peg_half_width]
        )
        
        # Add visual shapes with different colors
        # peg head (first color)
        mat1 = sapien.render.RenderMaterial()
        mat1.set_base_color(color_1)
        mat1.roughness = 0.5
        mat1.specular = 0.5
        builder.add_box_visual(
            sapien.Pose([self.peg_half_length / 2, 0, 0]),
            half_size=[self.peg_half_length / 2, self.peg_half_width, self.peg_half_width],
            material=mat1,
        )
        
        # peg tail (second color)
        mat2 = sapien.render.RenderMaterial()
        mat2.set_base_color(color_2)
        mat2.roughness = 0.5
        mat2.specular = 0.5
        builder.add_box_visual(
            sapien.Pose([-self.peg_half_length / 2, 0, 0]),
            half_size=[self.peg_half_length / 2, self.peg_half_width, self.peg_half_width],
            material=mat2,
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

        # load the two pegs with different colors
        self.red_blue_peg = self._build_peg(
            color_1=np.array([176, 14, 14, 255]) / 255,  # red
            color_2=np.array([12, 42, 160, 255]) / 255,  # blue
            name="red_blue_peg",
            initial_pos=[-0.08, -0.05, self.peg_half_width]
        )
        
        self.orange_white_peg = self._build_peg(
            color_1=sapien_utils.hex2rgba("#EC7357"),  # orange
            color_2=sapien_utils.hex2rgba("#EDF6F9"),  # white
            name="orange_white_peg",
            initial_pos=[-0.08, 0.05, self.peg_half_width]
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            # init the table scene
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            xyz = torch.zeros((b, 3))
            q = [1, 0, 0, 0]
            
            # Red-blue peg position (fixed)
            xyz[..., 0] = -0.08  # fixed x position
            xyz[..., 1] = -0.05  # fixed y position  
            xyz[..., 2] = self.peg_half_width  # on the table
            red_blue_peg_pose = Pose.create_from_pq(p=xyz.clone(), q=q)
            self.red_blue_peg.set_pose(red_blue_peg_pose)

            # Orange-white peg position (fixed, with guaranteed distance)
            xyz[..., 0] = -0.08  # same x as first peg
            xyz[..., 1] = 0.05   # fixed y position, 0.1 units away from first peg
            xyz[..., 2] = self.peg_half_width  # on the table
            orange_white_peg_pose = Pose.create_from_pq(p=xyz, q=q)
            self.orange_white_peg.set_pose(orange_white_peg_pose)

    def evaluate(self):
        # Check if pegs are grasped
        is_red_blue_peg_grasped = self.agent.is_grasping(self.red_blue_peg)
        is_orange_white_peg_grasped = self.agent.is_grasping(self.orange_white_peg)
        
        # Success: grasp the orange-white peg (the one with white color) - perception task only requires grasping
        success = is_orange_white_peg_grasped
        
        return {
            "is_red_blue_peg_grasped": is_red_blue_peg_grasped,
            "is_orange_white_peg_grasped": is_orange_white_peg_grasped,
            "success": success,
        }

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            is_red_blue_peg_grasped=info["is_red_blue_peg_grasped"],
            is_orange_white_peg_grasped=info["is_orange_white_peg_grasped"],
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )
        if "state" in self.obs_mode:
            obs.update(
                red_blue_peg_pose=self.red_blue_peg.pose.raw_pose,
                orange_white_peg_pose=self.orange_white_peg.pose.raw_pose,
                tcp_to_red_blue_peg_pos=self.red_blue_peg.pose.p - self.agent.tcp.pose.p,
                tcp_to_orange_white_peg_pos=self.orange_white_peg.pose.p - self.agent.tcp.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # reaching reward - reward for getting close to the ORANGE-WHITE PEG (the target)
        tcp_pose = self.agent.tcp.pose.p
        orange_white_peg_pos = self.orange_white_peg.pose.p
        
        obj_to_tcp_dist_orange_white = torch.linalg.norm(tcp_pose - orange_white_peg_pos, axis=1)
        reward = 2 * (1 - torch.tanh(5 * obj_to_tcp_dist_orange_white))

        # grasp reward - reward for grasping the correct (orange-white) peg
        is_orange_white_grasped = info["is_orange_white_peg_grasped"]
        reward[is_orange_white_grasped] = 5.0

        # success reward - successfully grasped the orange-white peg
        reward[info["success"]] = 10.0
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
        # this should be equal to compute_dense_reward / max possible reward
        max_reward = 10.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward