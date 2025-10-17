from typing import Any, Dict, Union

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import sapien
import torch
import torch.random
from transforms3d.euler import euler2quat
import os

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

@register_env("Dynamic-CatchRollingSphere-Hard", max_episode_steps=300)
class DynamicHardCatchRollingSphereEnv(BaseEnv):
    """
    **Task Description:**
    Catch the rolling sphere into the shallow bin, but only when the light turns green. The sphere continuously 
    rolls on the table, and the light bulb starts OFF and turns GREEN after 50 steps. The robot must wait for 
    the green light before attempting to grasp the sphere, otherwise the task fails.

    **Randomizations:**
    - The position of the bin and the sphere are randomized: The bin is initialized in [0, 0.1] x [-0.1, 0.1],
    and the sphere is initialized in [-0.1, -0.05] x [-0.1, 0.1]
    - The sphere starts with slow random rolling velocity and direction
    - Light bulb position is fixed on the table

    **Success Conditions:**
    - The light bulb shows GREEN light (after 50 steps)
    - The rolling sphere is caught on the top of the bin
    - The robot remains static and the gripper is not closed at the end state

    **Failure Conditions:**
    - Robot attempts to grasp sphere before light turns green
    - Light is still OFF when robot tries to manipulate sphere

    **Timing Challenge:**
    - Light starts OFF and turns GREEN after exactly 50 steps
    - Robot must wait and track the rolling sphere until permitted to act
    - Requires patience, timing, and dynamic object tracking
    """

    SUPPORTED_ROBOTS = ["panda", "fetch"]

    # Specify some supported robot types
    agent: Union[Panda, Fetch]

    # set some commonly used values
    radius = 0.02  # radius of the sphere
    inner_side_half_len = 0.02  # side length of the bin's inner square
    short_side_half_size = 0.0025  # length of the shortest edge of the block
    block_half_size = [
        short_side_half_size,
        2 * short_side_half_size + inner_side_half_len,
        2 * short_side_half_size + inner_side_half_len,
    ]  # The bottom block of the bin, which is larger: The list represents the half length of the block along the [x, y, z] axis respectively.
    edge_block_half_size = [
        short_side_half_size,
        2 * short_side_half_size + inner_side_half_len,
        2 * short_side_half_size,
    ]  # The edge block of the bin, which is smaller. The representations are similar to the above one
    
    max_rolling_speed = 0.05  # Reduced from 0.25 to make sphere move slowly
    LIGHT_ACTIVATION_STEPS = 25  # Light turns green after 50 steps

    LIGHT_BULB_OFF_PATH = "../../nebula/utils/building/assets/light_bulb/light_bulb_off.glb"
    LIGHT_BULB_GREEN_PATH = "../../nebula/utils/building/assets/light_bulb/light_bulb_green.glb"

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        
        # Light states
        self.LIGHT_OFF = 0
        self.LIGHT_GREEN = 1
        
        # Episode state variables
        self.current_light_state = self.LIGHT_OFF
        self.step_counter = 0
        self.light_activated = False
        self.episode_failed = False
        
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

    def _build_light_bulb(self, glb_file, name):
        """Build a light bulb actor from GLB file"""
        builder = self.scene.create_actor_builder()
        # print(f"Building light bulb: {name}")
        
        # Add collision to make it stable on table
        builder.add_box_collision(half_size=[0.02, 0.02, 0.03])
        
        # Load GLB file
        builder.add_visual_from_file(filename=glb_file, scale=[0.01, 0.01, 0.01])
        # print(f"Successfully loaded GLB: {glb_file}")
        
        # Set initial pose to prevent warning
        builder.initial_pose = sapien.Pose(p=[0.15, 0.0, 0.05])
        
        return builder.build_kinematic(name=name)

    def _build_bin(self, radius):
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

        # Set initial pose to prevent warning
        builder.initial_pose = sapien.Pose(p=[0, 0, 0])
        
        # build the kinematic bin
        return builder.build_kinematic(name="bin")

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        # load the table
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # load the rolling sphere
        self.obj = actors.build_sphere(
            self.scene,
            radius=self.radius,
            color=np.array([12, 42, 160, 255]) / 255,
            name="rolling_sphere",
            body_type="dynamic",
            initial_pose=Pose.create_from_pq(p=[0, 0, self.radius], q=[1, 0, 0, 0])
        )

        # load the bin
        self.bin = self._build_bin(self.radius)

        # Create light bulb actors at different positions
        self.light_off = self._build_light_bulb(self.LIGHT_BULB_OFF_PATH, "light_off")
        self.light_green = self._build_light_bulb(self.LIGHT_BULB_GREEN_PATH, "light_green")

    def _set_light_state(self, state):
        """Move the active light to visible position, others to hidden positions"""
        # print(f"Setting light state to: {state}")
        
        # Hidden position (far away)
        hidden_pose = Pose.create_from_pq(
            torch.tensor([[10, 10, 10]], device=self.device),
            torch.tensor([[1, 0, 0, 0]], device=self.device)
        )
        
        # Fixed visible position for light bulb
        visible_pose = Pose.create_from_pq(
            torch.tensor([[0.15, 0.0, 0.05]], device=self.device),  # Fixed position
            torch.tensor([euler2quat(np.pi/2, 0, 0)], device=self.device)  # Upright rotation
        )
        
        # Move all lights to hidden position first
        self.light_off.set_pose(hidden_pose)
        self.light_green.set_pose(hidden_pose)
        
        # Move the active light to visible position
        if state == self.LIGHT_OFF:
            self.light_off.set_pose(visible_pose)
        elif state == self.LIGHT_GREEN:
            self.light_green.set_pose(visible_pose)
        
        self.current_light_state = state

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            # init the table scene
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # Reset episode state
            self.current_light_state = self.LIGHT_OFF
            self.step_counter = 0
            self.light_activated = False
            self.episode_failed = False

            # init the sphere in the first 1/4 zone along the x-axis (so that it doesn't collide the bin)
            xyz = torch.zeros((b, 3))
            xyz[..., 0] = (torch.rand((b, 1)) * 0.05 - 0.1)[
                ..., 0
            ]  # first 1/4 zone of x ([-0.1, -0.05])
            xyz[..., 1] = (torch.rand((b, 1)) * 0.2 - 0.1)[
                ..., 0
            ]  # spanning all possible ys
            xyz[..., 2] = self.radius  # on the table
            q = [1, 0, 0, 0]
            obj_pose = Pose.create_from_pq(p=xyz, q=q)
            self.obj.set_pose(obj_pose)
            
            # Add slow rolling motion - random direction and reduced speed
            rolling_direction = torch.rand((b, 2)) * 2 - 1  # Random direction in xy plane
            rolling_direction = rolling_direction / torch.linalg.norm(rolling_direction, dim=1, keepdim=True)  # Normalize
            rolling_speed = torch.rand((b, 1)) * self.max_rolling_speed  # Slower random speed
            
            # Set linear velocity (slow rolling motion)
            linear_velocity = torch.zeros((b, 3))
            linear_velocity[:, :2] = rolling_direction * rolling_speed
            
            # Set angular velocity (for realistic rolling - v = ω × r)
            angular_velocity = torch.zeros((b, 3))
            # For rolling without slipping: ω = v/r, direction perpendicular to motion
            angular_velocity[:, 0] = -linear_velocity[:, 1] / self.radius  # ωx = -vy/r
            angular_velocity[:, 1] = linear_velocity[:, 0] / self.radius   # ωy = vx/r
            angular_velocity[:, 2] = 0  # No rotation around z-axis for pure rolling
            
            # Apply velocities to make sphere start rolling slowly
            self.obj.linear_velocity = linear_velocity
            self.obj.angular_velocity = angular_velocity

            # init the bin in the last 1/2 zone along the x-axis (so that it doesn't collide the sphere)
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

            # Set initial light state to OFF
            self._set_light_state(self.LIGHT_OFF)

    def step(self, action):
        """Override step to handle light activation and failure detection"""
        # Regular environment step
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Increment step counter
        self.step_counter += 1
        
        # Check if light should turn green
        if self.step_counter >= self.LIGHT_ACTIVATION_STEPS and not self.light_activated:
            self._set_light_state(self.LIGHT_GREEN)
            self.light_activated = True
            # print(f"Light activated at step {self.step_counter}!")
        
        # Check for failure condition: grasping before light is green
        if self.current_light_state == self.LIGHT_OFF and self.agent.is_grasping(self.obj).any():
            self.episode_failed = True
            print("FAILURE: Attempted to grasp sphere before light turned green!")
        
        return obs, reward, terminated, truncated, info

    def evaluate(self):
        pos_obj = self.obj.pose.p
        pos_bin = self.bin.pose.p
        offset = pos_obj - pos_bin
        xy_flag = torch.linalg.norm(offset[..., :2], axis=1) <= 0.005
        z_flag = (
            torch.abs(offset[..., 2] - self.radius - self.block_half_size[0]) <= 0.005
        )
        is_obj_on_bin = torch.logical_and(xy_flag, z_flag)
        is_obj_static = self.obj.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        is_obj_grasped = self.agent.is_grasping(self.obj)
        
        # Success only if light is green and task completed
        is_light_green = torch.tensor([self.current_light_state == self.LIGHT_GREEN], device=self.device)
        success = is_obj_on_bin & is_obj_static & (~is_obj_grasped) & is_light_green
        
        # Failure if grasped before light activation
        failed = torch.tensor([self.episode_failed], device=self.device)
        
        return {
            "is_obj_grasped": is_obj_grasped,
            "is_obj_on_bin": is_obj_on_bin,
            "is_obj_static": is_obj_static,
            "is_light_green": is_light_green,
            "failed": failed,
            "success": success,
            "step_counter": torch.tensor([self.step_counter], device=self.device),
        }

    def _get_obs_extra(self, info: Dict):
        # Include sphere velocity and light state in observations
        obs = dict(
            is_grasped=info["is_obj_grasped"],
            tcp_pose=self.agent.tcp.pose.raw_pose,
            bin_pos=self.bin.pose.p,
            sphere_velocity=self.obj.linear_velocity,  # Important for tracking rolling sphere
            sphere_angular_velocity=self.obj.angular_velocity,
            light_state=torch.tensor([self.current_light_state], device=self.device),
            step_counter=torch.tensor([self.step_counter], device=self.device),
            light_activated=torch.tensor([self.light_activated], device=self.device),
        )
        if "state" in self.obs_mode:
            obs.update(
                obj_pose=self.obj.pose.raw_pose,
                tcp_to_obj_pos=self.obj.pose.p - self.agent.tcp.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # Enhanced reward function with light state constraints
        reward = torch.zeros(self.num_envs, device=self.device)
        
        # Base reward for approaching sphere (but only when light is green)
        tcp_pose = self.agent.tcp.pose.p
        obj_pos = self.obj.pose.p
        obj_to_tcp_dist = torch.linalg.norm(tcp_pose - obj_pos, axis=1)
        
        if self.current_light_state == self.LIGHT_GREEN:
            # Normal reaching reward when light is green
            reaching_reward = 2 * (1 - torch.tanh(5 * obj_to_tcp_dist))
            reward += reaching_reward
        else:
            # Small penalty for getting too close when light is off (encourage patience)
            if obj_to_tcp_dist < 0.1:  # If too close to sphere when light is off
                reward -= 0.5

        # Waiting reward - encourage robot to wait patiently
        if self.current_light_state == self.LIGHT_OFF and not self.agent.is_grasping(self.obj).any():
            waiting_reward = 0.1  # Small reward for waiting
            reward += waiting_reward

        # Light activation bonus
        if self.light_activated and self.current_light_state == self.LIGHT_GREEN:
            reward += 1.0  # Bonus for light turning green

        # Grasp and place reward (only when light is green)
        if self.current_light_state == self.LIGHT_GREEN:
            bin_top_pos = self.bin.pose.p.clone()
            bin_top_pos[:, 2] = bin_top_pos[:, 2] + self.block_half_size[0] + self.radius
            obj_to_bin_top_dist = torch.linalg.norm(bin_top_pos - obj_pos, axis=1)
            place_reward = 1 - torch.tanh(5.0 * obj_to_bin_top_dist)
            
            # Higher reward for grasping a rolling sphere when allowed
            reward[info["is_obj_grasped"]] = (6 + place_reward)[info["is_obj_grasped"]]

            # ungrasp and static reward
            gripper_width = (self.agent.robot.get_qlimits()[0, -1, 1] * 2).to(self.device)
            is_obj_grasped = info["is_obj_grasped"]
            ungrasp_reward = (
                torch.sum(self.agent.robot.get_qpos()[:, -2:], axis=1) / gripper_width
            )
            ungrasp_reward[~is_obj_grasped] = 16.0
            v = torch.linalg.norm(self.obj.linear_velocity, axis=1)
            av = torch.linalg.norm(self.obj.angular_velocity, axis=1)
            static_reward = 1 - torch.tanh(v * 10 + av)
            robot_static_reward = self.agent.is_static(0.2)
            reward[info["is_obj_on_bin"]] = (
                8 + (ungrasp_reward + static_reward + robot_static_reward) / 3.0
            )[info["is_obj_on_bin"]]

        # Success reward (highest)
        if info["success"].any():
            reward += 20.0

        # Failure penalty (grasping before light activation)
        if info["failed"].any():
            reward -= 10.0

        # Small penalty if sphere rolls off table
        sphere_pos = self.obj.pose.p
        table_bounds = 0.4
        off_table_penalty = torch.where(
            (torch.abs(sphere_pos[:, 0]) > table_bounds) | 
            (torch.abs(sphere_pos[:, 1]) > table_bounds) |
            (sphere_pos[:, 2] < 0.5),
            -1.0, 0.0
        )
        reward += off_table_penalty

        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
        # this should be equal to compute_dense_reward / max possible reward
        max_reward = 20.0  # Updated max reward
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward