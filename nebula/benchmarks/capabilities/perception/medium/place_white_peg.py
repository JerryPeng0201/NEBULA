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


@register_env("Perception-PlaceWhitePeg-Medium", max_episode_steps=50)
class PerceptionPlaceWhitePegMediumEnv(BaseEnv):
    """
    **Task Description:**
    Place either peg (red-blue or orange-white) into the shallow bin.

    **Randomizations:**
    - The position of the bin and the pegs are randomized: The bin is initialized in [0, 0.1] x [-0.1, 0.1],
    and the pegs are initialized at fixed positions in [-0.1, -0.05] x [-0.1, 0.1]

    **Success Conditions:**
    - Either peg is placed on the top of the bin. The robot remains static and the gripper is not closed at the end state.
    """

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/PlaceSphere-v1_rt.mp4"
    SUPPORTED_ROBOTS = ["panda", "fetch"]

    # Specify some supported robot types
    agent: Union[Panda, Fetch]

    # set some commonly used values for pegs
    peg_half_length = 0.05  # half length of the peg
    peg_half_width = 0.02  # half width of the peg cross-section
    peg_height = 0.02  # height of the peg (2 * peg_half_width)
    
    # bin dimensions adjusted for pegs
    inner_side_half_len = 0.06  # increased to fit peg length
    short_side_half_size = 0.0025   # increased thickness
    block_half_size = [
        short_side_half_size,
        2 * short_side_half_size + inner_side_half_len,
        2 * short_side_half_size + inner_side_half_len,
    ]
    edge_block_half_size = [
        short_side_half_size,
        2 * short_side_half_size + inner_side_half_len,
        2 * short_side_half_size,
    ]

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

    def _build_bin(self):
        builder = self.scene.create_actor_builder()

        # init the locations of the basic blocks
        dx = self.block_half_size[1] - self.block_half_size[0]
        dy = self.block_half_size[1] - self.block_half_size[0]
        dz = self.edge_block_half_size[2] + self.block_half_size[0]

        # build the bin bottom and edge blocks
        poses = [
            sapien.Pose([0, 0, 0]),
            sapien.Pose([-dx, 0, dz]),
            sapien.Pose([dx, 0, dz]),
            sapien.Pose([0, -dy, dz]),
            sapien.Pose([0, dy, dz]),
        ]
        half_sizes = [
            [self.block_half_size[1], self.block_half_size[2], self.block_half_size[0]],
            self.edge_block_half_size,
            self.edge_block_half_size,
            [
                self.edge_block_half_size[1],
                self.edge_block_half_size[0],
                self.edge_block_half_size[2],
            ],
            [
                self.edge_block_half_size[1],
                self.edge_block_half_size[0],
                self.edge_block_half_size[2],
            ],
        ]
        for pose, half_size in zip(poses, half_sizes):
            builder.add_box_collision(pose, half_size)
            builder.add_box_visual(pose, half_size)

        # build the kinematic bin
        return builder.build_kinematic(name="bin")

    def _build_peg(self, color_1, color_2, name):
        """Build a two-color peg"""
        builder = self.scene.create_actor_builder()
        
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
            name="red_blue_peg"
        )
        
        self.orange_white_peg = self._build_peg(
            color_1=sapien_utils.hex2rgba("#EC7357"),  # orange
            color_2=sapien_utils.hex2rgba("#EDF6F9"),  # white
            name="orange_white_peg"
        )

        # load the bin
        self.bin = self._build_bin()

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

            # init the bin in the last 1/2 zone along the x-axis (so that it doesn't collide the pegs)
            pos = torch.zeros((b, 3))
            pos[:, 0] = (
                torch.rand((b, 1))[..., 0] * 0.1
            )  # the last 1/2 zone of x ([0, 0.1])
            pos[:, 1] = (
                torch.rand((b, 1))[..., 0] * 0.2 - 0.1
            )  # spanning all possible ys
            pos[:, 2] = self.block_half_size[0]  # on the table
            q = [1, 0, 0, 0]
            bin_pose = Pose.create_from_pq(p=pos, q=q)
            self.bin.set_pose(bin_pose)

    def evaluate(self):
        pos_red_blue_peg = self.red_blue_peg.pose.p
        pos_orange_white_peg = self.orange_white_peg.pose.p
        pos_bin = self.bin.pose.p
        
        # Check if either peg is on the bin
        is_red_blue_peg_on_bin = self._check_peg_on_bin(pos_red_blue_peg, pos_bin)
        is_orange_white_peg_on_bin = self._check_peg_on_bin(pos_orange_white_peg, pos_bin)
        is_any_peg_on_bin = torch.logical_or(is_red_blue_peg_on_bin, is_orange_white_peg_on_bin)
        
        # Check if pegs are static
        is_red_blue_peg_static = self.red_blue_peg.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        is_orange_white_peg_static = self.orange_white_peg.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        
        # Check if pegs are grasped
        is_red_blue_peg_grasped = self.agent.is_grasping(self.red_blue_peg)
        is_orange_white_peg_grasped = self.agent.is_grasping(self.orange_white_peg)
        
        # Success: either peg on bin, that peg is static, and that peg is not grasped
        red_blue_peg_success = is_red_blue_peg_on_bin & is_red_blue_peg_static & (~is_red_blue_peg_grasped)
        orange_white_peg_success = is_orange_white_peg_on_bin & is_orange_white_peg_static & (~is_orange_white_peg_grasped)
        success = torch.logical_or(red_blue_peg_success, orange_white_peg_success)
        
        return {
            "is_red_blue_peg_grasped": is_red_blue_peg_grasped,
            "is_orange_white_peg_grasped": is_orange_white_peg_grasped,
            "is_red_blue_peg_on_bin": is_red_blue_peg_on_bin,
            "is_orange_white_peg_on_bin": is_orange_white_peg_on_bin,
            "is_red_blue_peg_static": is_red_blue_peg_static,
            "is_orange_white_peg_static": is_orange_white_peg_static,
            "success": success,
        }

    def _check_peg_on_bin(self, pos_peg, pos_bin):
        """Check if peg is on top of the bin"""
        offset = pos_peg - pos_bin
        # More lenient xy check for peg placement (accounting for peg length)
        xy_flag = torch.linalg.norm(offset[..., :2], axis=1) <= 0.015
        z_flag = (
            torch.abs(offset[..., 2] - self.peg_half_width - self.block_half_size[0]) <= 0.01
        )
        return torch.logical_and(xy_flag, z_flag)

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            is_red_blue_peg_grasped=info["is_red_blue_peg_grasped"],
            is_orange_white_peg_grasped=info["is_orange_white_peg_grasped"],
            tcp_pose=self.agent.tcp.pose.raw_pose,
            bin_pos=self.bin.pose.p,
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
        # reaching reward - reward for getting close to either peg
        tcp_pose = self.agent.tcp.pose.p
        red_blue_peg_pos = self.red_blue_peg.pose.p
        orange_white_peg_pos = self.orange_white_peg.pose.p
        
        obj_to_tcp_dist_red_blue = torch.linalg.norm(tcp_pose - red_blue_peg_pos, axis=1)
        obj_to_tcp_dist_orange_white = torch.linalg.norm(tcp_pose - orange_white_peg_pos, axis=1)
        obj_to_tcp_dist = torch.minimum(obj_to_tcp_dist_red_blue, obj_to_tcp_dist_orange_white)
        reward = 2 * (1 - torch.tanh(5 * obj_to_tcp_dist))

        # grasp and place reward
        bin_top_pos = self.bin.pose.p.clone()
        bin_top_pos[:, 2] = bin_top_pos[:, 2] + self.block_half_size[0] + self.peg_half_width
        
        red_blue_peg_to_bin_dist = torch.linalg.norm(bin_top_pos - red_blue_peg_pos, axis=1)
        orange_white_peg_to_bin_dist = torch.linalg.norm(bin_top_pos - orange_white_peg_pos, axis=1)
        
        # Reward based on whichever peg is grasped
        is_red_blue_grasped = info["is_red_blue_peg_grasped"]
        is_orange_white_grasped = info["is_orange_white_peg_grasped"]
        
        red_blue_place_reward = 1 - torch.tanh(5.0 * red_blue_peg_to_bin_dist)
        orange_white_place_reward = 1 - torch.tanh(5.0 * orange_white_peg_to_bin_dist)
        
        reward[is_red_blue_grasped] = (4 + red_blue_place_reward)[is_red_blue_grasped]
        reward[is_orange_white_grasped] = (4 + orange_white_place_reward)[is_orange_white_grasped]

        # ungrasp and static reward
        gripper_width = (self.agent.robot.get_qlimits()[0, -1, 1] * 2).to(self.device)
        is_any_peg_grasped = torch.logical_or(is_red_blue_grasped, is_orange_white_grasped)
        ungrasp_reward = (
            torch.sum(self.agent.robot.get_qpos()[:, -2:], axis=1) / gripper_width
        )
        ungrasp_reward[
            ~is_any_peg_grasped
        ] = 16.0  # give ungrasp a bigger reward
        
        # Static reward for whichever peg is on the bin
        red_blue_v = torch.linalg.norm(self.red_blue_peg.linear_velocity, axis=1)
        red_blue_av = torch.linalg.norm(self.red_blue_peg.angular_velocity, axis=1)
        red_blue_static_reward = 1 - torch.tanh(red_blue_v * 10 + red_blue_av)
        
        orange_white_v = torch.linalg.norm(self.orange_white_peg.linear_velocity, axis=1)
        orange_white_av = torch.linalg.norm(self.orange_white_peg.angular_velocity, axis=1)
        orange_white_static_reward = 1 - torch.tanh(orange_white_v * 10 + orange_white_av)
        
        robot_static_reward = self.agent.is_static(0.2)
        
        is_red_blue_on_bin = info["is_red_blue_peg_on_bin"]
        is_orange_white_on_bin = info["is_orange_white_peg_on_bin"]
        is_any_on_bin = torch.logical_or(is_red_blue_on_bin, is_orange_white_on_bin)
        
        # Use the static reward for whichever peg is on the bin
        static_reward = torch.where(is_red_blue_on_bin, red_blue_static_reward, orange_white_static_reward)
        
        reward[is_any_on_bin] = (
            6 + (ungrasp_reward + static_reward + robot_static_reward) / 3.0
        )[is_any_on_bin]

        # success reward
        reward[info["success"]] = 13
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
        # this should be equal to compute_dense_reward / max possible reward
        max_reward = 13.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward