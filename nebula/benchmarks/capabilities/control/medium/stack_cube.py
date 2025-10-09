from typing import Any, Dict, Union

import numpy as np
import sapien
import torch

from nebula.core.embodiment.robots import Fetch, Panda
from nebula.core.simulation.engine import BaseEnv
from nebula.core.simulation.utils import randomization
from nebula.utils.geometry.rotation_conversions import (
    euler_angles_to_matrix,
    matrix_to_quaternion,
)
from nebula.core.sensors.camera import CameraConfig
from nebula.utils import common, sapien_utils
from nebula.utils.building import actors
from nebula.utils.registration import register_env
from nebula.utils.scene_builder.table import TableSceneBuilder
from nebula.utils.structs.pose import Pose  
from nebula.utils.logging_utils import logger

@register_env("Control-StackCube-Medium", max_episode_steps=250)
class ControlStackCubeMediumEnv(BaseEnv):
    """
    **Task Description:**
    - The goal is to pick up a red cube, place it next to the green cube, and stack the blue cube on top of the red and green cube without it falling off.

    **Randomizations:**
    - all cubes have their z-axis rotation randomized
    - all cubes have their xy positions on top of the table scene randomized. The positions are sampled such that the cubes do not collide with each other

    **Success Conditions:**
    - the blue cube is static
    - the blue cube is on top of both the red and green cube (to within half of the cube size)
    - none of the red, green, blue cubes are grasped by the robot (robot must let go of the cubes)

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/StackPyramid-v1_rt.mp4"

    """

    SUPPORTED_ROBOTS = ["panda", "fetch"]
    SUPPORTED_REWARD_MODES = ["none", "sparse"]
    
    agent: Union[Panda, Fetch]

    def __init__(
        self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

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
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)


    def _load_scene(self, options: dict):
        self.cube_half_size = common.to_tensor([0.02] * 3)
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.cubeA = actors.build_cube(
            self.scene, half_size=0.02, color=[1, 0, 0, 1], name="cubeA", initial_pose=sapien.Pose(p=[0, 0, 0.2])
        )
        self.cubeB = actors.build_cube(
            self.scene, half_size=0.02, color=[0, 1, 0, 1], name="cubeB", initial_pose=sapien.Pose(p=[1, 0, 0.2])
        )
        self.cubeC = actors.build_cube(
            self.scene, half_size=0.02, color=[0, 0, 1, 1], name="cubeC", initial_pose=sapien.Pose(p=[-1, 0, 0.2])
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            xyz = torch.zeros((b, 3), device=self.device)
            xyz[:, 2] = 0.02
            
            # Fixed positions for the three cubes
            cubeA_xy = torch.zeros((b, 2), device=self.device)
            cubeA_xy[:, 0] = -0.1  # fixed x position for cubeA
            cubeA_xy[:, 1] = -0.1  # fixed y position for cubeA
            
            cubeB_xy = torch.zeros((b, 2), device=self.device)
            cubeB_xy[:, 0] = 0.0    # fixed x position for cubeB
            cubeB_xy[:, 1] = 0.0    # fixed y position for cubeB
            
            cubeC_xy = torch.zeros((b, 2), device=self.device)
            cubeC_xy[:, 0] = 0.1   # fixed x position for cubeC
            cubeC_xy[:, 1] = 0.1   # fixed y position for cubeC

            # Cube A
            xyz[:, :2] = cubeA_xy
            qs = randomization.random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=False,
            )
            self.cubeA.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs))

            # Cube B
            xyz[:, :2] = cubeB_xy
            qs = randomization.random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=False,
            )
            self.cubeB.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs))
            
            # Cube C
            xyz[:, :2] = cubeC_xy
            qs = randomization.random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=False,
            )
            self.cubeC.set_pose(Pose.create_from_pq(p=xyz, q=qs))

    def evaluate(self):
        pos_A = self.cubeA.pose.p
        pos_B = self.cubeB.pose.p
        pos_C = self.cubeC.pose.p

        offset_AB = pos_A - pos_B
        offset_BC = pos_B - pos_C
        offset_AC = pos_A - pos_C

        def evaluate_cube_distance(offset, cube_a, cube_b, top_or_next):
            xy_flag = (torch.linalg.norm(offset[..., :2], axis=1) 
                       <= torch.linalg.norm(2*self.cube_half_size[:2]) 
                       + 0.005
                       )
            z_flag = torch.abs(offset[..., 2]) > 0.02
            if top_or_next == "top":
                is_cubeA_on_cubeB = torch.logical_and(xy_flag, z_flag)
            elif top_or_next == "next_to":
                is_cubeA_on_cubeB = xy_flag
            else:
                return NotImplementedError(f"Expect top_or_next to be either 'top' or 'next_to', got {top_or_next}")
            
            is_cubeA_static = cube_a.is_static(lin_thresh=1e-2, ang_thresh=0.5)
            is_cubeA_grasped = self.agent.is_grasping(cube_a)

            success = is_cubeA_on_cubeB & is_cubeA_static & (~is_cubeA_grasped)            
            return success.bool()

        success_A_B = evaluate_cube_distance(offset_AB, self.cubeA, self.cubeB, "next_to")
        success_C_B = evaluate_cube_distance(offset_BC, self.cubeC, self.cubeB, "top")
        success_C_A = evaluate_cube_distance(offset_AC, self.cubeC, self.cubeA, "top")
        success = torch.logical_and(success_A_B, torch.logical_and(success_C_B, success_C_A))
        return {
            "success": success,
        }

    def _get_obs_extra(self, info: Dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if "state" in self.obs_mode:
            obs.update(
                cubeA_pose=self.cubeA.pose.raw_pose,
                cubeB_pose=self.cubeB.pose.raw_pose,
                cubeC_pose=self.cubeC.pose.raw_pose,
                tcp_to_cubeA_pos=self.cubeA.pose.p - self.agent.tcp.pose.p,
                tcp_to_cubeB_pos=self.cubeB.pose.p - self.agent.tcp.pose.p,
                tcp_to_cubeC_pos=self.cubeC.pose.p - self.agent.tcp.pose.p,
                cubeA_to_cubeB_pos=self.cubeB.pose.p - self.cubeA.pose.p,
                cubeB_to_cubeC_pos=self.cubeC.pose.p - self.cubeB.pose.p,
                cubeA_to_cubeC_pos=self.cubeC.pose.p - self.cubeA.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Dict, action: torch.Tensor, info: Dict):
        
        reward = torch.zeros(self.num_envs, device=self.device)
        
        # Get cube positions
        pos_A = self.cubeA.pose.p  # Red cube
        pos_B = self.cubeB.pose.p  # Green cube  
        pos_C = self.cubeC.pose.p  # Blue cube
        tcp_pos = self.agent.tcp.pose.p
        
        # Calculate key distances
        dist_AB_xy = torch.linalg.norm(pos_A[:, :2] - pos_B[:, :2], axis=1)  # Red-green horizontal distance
        dist_AC = torch.linalg.norm(pos_A - pos_C, axis=1)  # Red-blue distance
        dist_BC = torch.linalg.norm(pos_B - pos_C, axis=1)  # Green-blue distance
        
        # Target distances for evaluation
        target_adjacent_dist = torch.linalg.norm(2 * self.cube_half_size[:2])  # For side-by-side placement
        
        # 1. Reaching rewards - encourage approaching cubes in task order
        tcp_to_cubes = {
            'A': torch.linalg.norm(tcp_pos - pos_A, axis=1),  # Distance to red cube
            'B': torch.linalg.norm(tcp_pos - pos_B, axis=1),  # Distance to green cube
            'C': torch.linalg.norm(tcp_pos - pos_C, axis=1)   # Distance to blue cube
        }
        
        # Phase-based reaching reward
        # First priority: Get red cube next to green cube
        red_green_adjacent = dist_AB_xy < (target_adjacent_dist + 0.01)
        
        if not red_green_adjacent.any():
            # Focus on red and green cubes first
            reaching_reward = (1.0 - torch.tanh(3.0 * tcp_to_cubes['A'])) * 0.7
            reaching_reward += (1.0 - torch.tanh(4.0 * tcp_to_cubes['B'])) * 0.3  # Green is stationary reference
            reward += reaching_reward
        else:
            # Once red-green are adjacent, focus on blue cube for stacking
            reaching_reward = (1.0 - torch.tanh(3.0 * tcp_to_cubes['C'])) * 1.0
            reward += reaching_reward
        
        # 2. Grasping rewards
        is_grasping_A = self.agent.is_grasping(self.cubeA)
        is_grasping_B = self.agent.is_grasping(self.cubeB)
        is_grasping_C = self.agent.is_grasping(self.cubeC)
        
        grasping_reward = (is_grasping_A.float() + is_grasping_B.float() + is_grasping_C.float()) * 1.5
        # Limit to grasping one cube at a time
        grasping_reward = torch.clamp(grasping_reward, 0, 1.5)
        reward += grasping_reward
        
        # 3. Red-Green adjacency reward
        # Encourage red cube to be placed next to green cube
        adjacency_reward = 1.0 - torch.tanh(5.0 * (dist_AB_xy - target_adjacent_dist))
        adjacency_reward = torch.clamp(adjacency_reward, 0, 1) * 2.5
        
        # Height alignment for red and green (should be at same level)
        height_diff_AB = torch.abs(pos_A[:, 2] - pos_B[:, 2])
        height_alignment_reward = (1.0 - torch.tanh(10.0 * height_diff_AB)) * 0.5
        
        reward += adjacency_reward + height_alignment_reward
        
        # 4. Blue cube stacking rewards
        # Blue cube should be positioned above both red and green cubes
        
        # Check if blue is above both red and green
        blue_above_red = pos_C[:, 2] > pos_A[:, 2] + 0.01
        blue_above_green = pos_C[:, 2] > pos_B[:, 2] + 0.01
        blue_above_both = blue_above_red & blue_above_green
        
        # XY positioning rewards - blue should be between/over red and green
        # Target position for blue cube (midpoint between red and green)
        target_blue_xy = (pos_A[:, :2] + pos_B[:, :2]) / 2.0
        blue_xy_error = torch.linalg.norm(pos_C[:, :2] - target_blue_xy, axis=1)
        blue_xy_reward = 1.0 - torch.tanh(5.0 * blue_xy_error)
        
        # Z positioning reward - blue should be exactly one cube height above base
        target_blue_z = pos_A[:, 2] + 2 * self.cube_half_size[2]  # One cube height above red/green
        blue_z_error = torch.abs(pos_C[:, 2] - target_blue_z)
        blue_z_reward = 1.0 - torch.tanh(8.0 * blue_z_error)
        
        # Only give stacking rewards when blue is above both cubes
        stacking_reward = torch.zeros_like(reward)
        stacking_reward[blue_above_both] = (blue_xy_reward + blue_z_reward)[blue_above_both] * 1.5
        
        # Additional reward for blue cube being close to both red and green
        blue_covers_red = torch.linalg.norm(pos_C[:, :2] - pos_A[:, :2], axis=1) < (target_adjacent_dist * 0.7)
        blue_covers_green = torch.linalg.norm(pos_C[:, :2] - pos_B[:, :2], axis=1) < (target_adjacent_dist * 0.7)
        coverage_bonus = (blue_covers_red & blue_covers_green & blue_above_both).float() * 1.0
        
        reward += stacking_reward + coverage_bonus
        
        # 5. Stability rewards
        stability_reward = torch.zeros_like(reward)
        
        # Individual cube stability
        cubes = [self.cubeA, self.cubeB, self.cubeC]
        for cube in cubes:
            is_static = cube.is_static(lin_thresh=1e-2, ang_thresh=0.5)
            is_not_grasped = ~self.agent.is_grasping(cube)
            stable = is_static & is_not_grasped
            stability_reward += stable.float() * 0.3
        
        reward += stability_reward
        
        # 6. Task-specific completion rewards
        completion_reward = torch.zeros_like(reward)
        
        # Red-green adjacency completion
        red_green_success = (dist_AB_xy <= target_adjacent_dist + 0.005) & \
                        (height_diff_AB < 0.01) & \
                        ~is_grasping_A & ~is_grasping_B & \
                        self.cubeA.is_static(lin_thresh=1e-2, ang_thresh=0.5) & \
                        self.cubeB.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        
        completion_reward += red_green_success.float() * 3.0
        
        # Blue stacking completion
        blue_on_red = (torch.linalg.norm(pos_A[:, :2] - pos_C[:, :2], axis=1) <= target_adjacent_dist + 0.005) & \
                    blue_above_red & (blue_z_error < 0.01)
        blue_on_green = (torch.linalg.norm(pos_B[:, :2] - pos_C[:, :2], axis=1) <= target_adjacent_dist + 0.005) & \
                        blue_above_green & (blue_z_error < 0.01)
        
        blue_stacking_success = blue_on_red & blue_on_green & ~is_grasping_C & \
                            self.cubeC.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        
        completion_reward += blue_stacking_success.float() * 4.0
        
        # Overall task success bonus
        overall_success = red_green_success & blue_stacking_success
        completion_reward += overall_success.float() * 8.0
        
        reward += completion_reward
        
        # 7. Penalty for undesirable behaviors
        penalty = torch.zeros_like(reward)
        
        # Penalize cubes falling off table or going too far
        for cube in cubes:
            cube_pos = cube.pose.p
            too_low = cube_pos[:, 2] < 0.005
            too_far = torch.linalg.norm(cube_pos[:, :2], axis=1) > 0.4
            penalty[too_low | too_far] += -2.0
        
        # Penalize grasping multiple cubes
        multiple_grasped = (is_grasping_A.float() + is_grasping_B.float() + is_grasping_C.float()) > 1
        penalty[multiple_grasped] += -1.0
        
        # Penalize blue cube being too far from the stacking area when red-green are ready
        if red_green_success.any():
            blue_too_far = torch.linalg.norm(pos_C[:, :2] - target_blue_xy, axis=1) > 0.15
            penalty[red_green_success & blue_too_far] += -0.5
        
        reward += penalty
        
        # 8. Progress tracking and momentum rewards
        if hasattr(self, '_prev_distances'):
            prev_dist_AB, prev_blue_xy_error, prev_blue_z_error = self._prev_distances
            
            # Reward progress in red-green positioning
            ab_progress = (prev_dist_AB - dist_AB_xy) * 3.0
            ab_progress = torch.clamp(ab_progress, -0.3, 0.8)
            
            # Reward progress in blue positioning (only when red-green are close)
            if red_green_adjacent.any():
                xy_progress = (prev_blue_xy_error - blue_xy_error) * 2.0
                z_progress = (prev_blue_z_error - blue_z_error) * 2.0
                blue_progress = torch.clamp(xy_progress + z_progress, -0.3, 1.0)
                reward[red_green_adjacent] += blue_progress[red_green_adjacent]
            
            reward += ab_progress
        
        # Store current distances for next iteration
        self._prev_distances = (dist_AB_xy.detach().clone(), 
                            blue_xy_error.detach().clone(),
                            blue_z_error.detach().clone())
        
        # 9. Exploration bonus for trying different approaches
        exploration_bonus = torch.zeros_like(reward)
        
        # Small bonus for moving cubes (encourages exploration)
        for cube in cubes:
            if hasattr(cube, 'linear_velocity'):
                velocity = torch.linalg.norm(cube.linear_velocity, axis=1)
                moving = (velocity > 0.01) & (velocity < 0.5)  # Not too fast, not too slow
                exploration_bonus += moving.float() * 0.1
        
        reward += exploration_bonus
        
        return reward

    def compute_normalized_dense_reward(self, obs: Dict, action: torch.Tensor, info: Dict):

        dense_reward = self.compute_dense_reward(obs, action, info)
        max_reward = 26.5
        return torch.clamp(dense_reward / max_reward, 0.0, 1.0)