
from typing import Any, Dict, Union

import numpy as np
import sapien
import torch

from nebula.core.embodiment.robots import Fetch, Panda
from nebula.core.simulation.engine import BaseEnv
from nebula.core.sensors.camera import CameraConfig
from nebula.utils import sapien_utils
from nebula.utils.building import actors
from nebula.utils.registration import register_env
from nebula.utils.scene_builder.table import TableSceneBuilder
from nebula.utils.structs.pose import Pose
from nebula.utils.structs.types import Array, GPUMemoryConfig, SimConfig

@register_env("AdaptationTest-MovingCube", max_episode_steps=100)
class AdaptationTestMovingCubeEnv(BaseEnv):
    """
    **Task Description:**
    Test adaptation speed when target object moves during execution.
    - A cube is randomly placed on the table
    - After 20 steps, the cube moves slightly to a new position
    - Goal: Successfully lift the cube 5cm above the table
    
    **Randomizations:**
    - Initial cube position is randomized within reachable area
    - Movement direction and distance are randomized (within limits)
    
    **Success Conditions:**
    - Cube is grasped by the robot
    - Cube is lifted at least 5cm above its starting height
    - Robot is relatively static (not moving wildly)
    """
    
    SUPPORTED_ROBOTS = ["panda", "fetch"]
    SUPPORTED_REWARD_MODES = ["none", "sparse", "dense"]
    
    agent: Union[Panda, Fetch]
    
    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.cube_half_size = 0.025  # 2.5cm half size = 5cm cube
        self.lift_height = 0.05  # 5cm lift requirement
        self.move_trigger_step = 20  # Move cube after 20 steps
        self.move_distance = 0.08  # Maximum movement distance (8cm)
        
        super().__init__(*args, robot_uids=robot_uids, **kwargs)
    
    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18
            )
        )
    
    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, -0.2, 0.2], [0.0, 0.0, 0.2])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
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
    
    def _load_scene(self, options: dict):
        # Build table
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        
        # Create target cube
        self.cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[0.8, 0.2, 0.2, 1],  # Red cube
            name="target_cube",
            initial_pose=sapien.Pose(p=[0, 0, 0.5])  # Start high, will reset in initialize
        )
        
        # Initialize tracking variables
        self.episode_step_count = 0
        self.cube_has_moved = False
        self.original_cube_pos = None
        self.new_cube_pos = None
        self.movement_triggered_at = -1
    
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            
            # Reset episode tracking
            self.episode_step_count = 0
            self.cube_has_moved = False
            self.movement_triggered_at = -1
            
            # Random initial position for cube (within robot's reach)
            cube_xyz = torch.zeros((b, 3), device=self.device)
            # Randomize x,y position within reasonable reach
            cube_xyz[:, 0] = torch.rand((b,), device=self.device) * 0.3 - 0.05  # [-0.05, 0.25]
            cube_xyz[:, 1] = torch.rand((b,), device=self.device) * 0.3 - 0.15  # [-0.15, 0.15]
            cube_xyz[:, 2] = self.cube_half_size  # On table surface
            
            # Set initial cube pose
            self.cube.set_pose(Pose.create_from_pq(p=cube_xyz))
            
            # Store original position for later movement
            self.original_cube_pos = cube_xyz.clone()
            
            # Pre-calculate where cube will move to (but don't move it yet)
            move_direction = torch.randn((b, 2), device=self.device)
            move_direction = move_direction / torch.linalg.norm(move_direction, dim=1, keepdim=True)
            move_distance = torch.rand((b, 1), device=self.device) * self.move_distance * 0.5 + self.move_distance * 0.5  # [0.04, 0.08]
            
            self.planned_movement = torch.zeros((b, 3), device=self.device)
            self.planned_movement[:, :2] = move_direction * move_distance
            
            # Ensure new position stays within reachable bounds
            new_pos = self.original_cube_pos + self.planned_movement
            new_pos[:, 0] = torch.clamp(new_pos[:, 0], -0.1, 0.3)
            new_pos[:, 1] = torch.clamp(new_pos[:, 1], -0.2, 0.2)
            self.new_cube_pos = new_pos
    
    def step(self, action):
        """Override step to implement cube movement after 20 steps"""
        # Call parent step
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Increment step counter
        self.episode_step_count += 1
        
        # Move cube at the specified step
        if self.episode_step_count == self.move_trigger_step and not self.cube_has_moved:
            self._move_cube()
            self.cube_has_moved = True
            self.movement_triggered_at = self.episode_step_count
            
            # Add movement info to the info dict
            info["cube_moved"] = True
            info["movement_step"] = self.episode_step_count
            info["movement_distance"] = torch.linalg.norm(self.planned_movement[:, :2], dim=1)
        
        return obs, reward, terminated, truncated, info
    
    def _move_cube(self):
        """Move the cube to its new position"""
        if self.new_cube_pos is not None:
            # Only move if cube is not currently grasped
            if not self.agent.is_grasping(self.cube).any():
                self.cube.set_pose(Pose.create_from_pq(p=self.new_cube_pos))
                # Small velocity to make movement more natural
                small_velocity = (self.new_cube_pos - self.original_cube_pos) * 2
                small_velocity[:, 2] = 0  # No vertical velocity
                self.cube.set_linear_velocity(small_velocity)
                
                # print(f"Cube moved at step {self.episode_step_count}")
    
    def evaluate(self):
        """Check if task is successfully completed"""
        cube_pos = self.cube.pose.p
        
        # Check if cube is grasped
        is_grasped = self.agent.is_grasping(self.cube)
        
        # Check if cube is lifted (5cm above table)
        is_lifted = cube_pos[:, 2] > (self.cube_half_size + self.lift_height)
        
        # Check if robot is relatively stable
        is_robot_static = self.agent.is_static(0.2)
        
        # Success: grasped, lifted, and stable
        success = is_grasped & is_lifted & is_robot_static
        
        return {
            "success": success,
            "is_grasped": is_grasped,
            "is_lifted": is_lifted,
            "is_robot_static": is_robot_static,
            "cube_has_moved": self.cube_has_moved,
            "steps_since_movement": self.episode_step_count - self.movement_triggered_at if self.cube_has_moved else -1,
            "adaptation_success": success & self.cube_has_moved,  # Success after cube moved
        }
    
    def _get_obs_extra(self, info: Dict):
        """Provide observation data"""
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            cube_pose=self.cube.pose.raw_pose,
            tcp_to_cube_pos=self.cube.pose.p - self.agent.tcp.pose.p,
            is_grasping=self.agent.is_grasping(self.cube),
            episode_step=torch.tensor(self.episode_step_count, device=self.device),
            cube_has_moved=torch.tensor(self.cube_has_moved, device=self.device),
        )
        
        if "state" in self.obs_mode:
            obs.update(
                cube_vel=self.cube.get_linear_velocity(),
                cube_ang_vel=self.cube.get_angular_velocity(),
                original_cube_pos=self.original_cube_pos if self.original_cube_pos is not None else torch.zeros((1, 3), device=self.device),
                planned_new_pos=self.new_cube_pos if self.new_cube_pos is not None else torch.zeros((1, 3), device=self.device),
            )
        
        return obs
    
    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        """Compute dense reward to guide learning"""
        reward = torch.zeros(self.num_envs, device=self.device)
        
        cube_pos = self.cube.pose.p
        tcp_pos = self.agent.tcp.pose.p
        
        # 1. Reaching reward (approach cube)
        tcp_to_cube_dist = torch.linalg.norm(cube_pos - tcp_pos, dim=1)
        reaching_reward = 1.0 - torch.tanh(5.0 * tcp_to_cube_dist)
        reward += reaching_reward
        
        # 2. Grasping reward
        is_grasping = self.agent.is_grasping(self.cube).float()
        reward += is_grasping * 2.0
        
        # 3. Lifting reward
        lift_progress = (cube_pos[:, 2] - self.cube_half_size) / self.lift_height
        lift_progress = torch.clamp(lift_progress, 0, 1)
        reward += lift_progress * 3.0
        
        # 4. Adaptation bonus (extra reward for succeeding after cube moves)
        if self.cube_has_moved:
            # Bonus for re-approaching cube after movement
            steps_since_move = self.episode_step_count - self.movement_triggered_at
            if steps_since_move < 20:  # Quick adaptation
                adaptation_bonus = (20 - steps_since_move) / 20.0
                reward += adaptation_bonus * is_grasping
        
        # 5. Success bonus
        if "success" in info:
            reward += info["success"].float() * 10.0
            
            # Extra bonus for adapting after movement
            if "adaptation_success" in info:
                reward += info["adaptation_success"].float() * 5.0
        
        return reward
    
    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        """Normalized reward in [0, 1]"""
        dense_reward = self.compute_dense_reward(obs, action, info)
        return torch.clamp(dense_reward / 20.0, 0.0, 1.0)