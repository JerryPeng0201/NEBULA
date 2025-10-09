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

# Control-StackCube-Hard

@register_env("Robust-StackCube-Easy", max_episode_steps=250)
class RobustStackCubeEasyEnv(BaseEnv):
    """
    **Task Description:**
    - The goal is to arrange four colored cubes: place the red cube next to the green cube, stack the blue cube on top of both red and green cubes, and place the purple cube next to the arrangement.

    **Randomizations:**
    - all cubes have fixed positions (no randomization)
    - all cubes have fixed orientations (no z-axis rotation randomization)

    **Success Conditions:**
    - the blue cube is static and on top of both the red and green cube
    - the purple cube is placed next to the arrangement
    - none of the cubes are grasped by the robot (robot must let go of the cubes)

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
    

    def _add_ycb_distractor_to_pushcube(self, env_idx: torch.Tensor):
        """
        Add 1-2 YCB distractor objects to the StackCube environment
        
        Args:
            env_idx: Environment indices to initialize
        """
        import random
        import os
        from nebula import ASSET_DIR
        from nebula.utils import assets, download_asset
        from nebula.utils.building import actors
        from nebula.utils.structs.pose import Pose
        
        # Clean up previous distractor objects if they exist
        if hasattr(self, 'distractor_objs'):
            for obj in self.distractor_objs:
                if obj is not None:
                    try:
                        self.scene.remove_actor(obj)
                    except:
                        pass
            self.distractor_objs = []
        else:
            self.distractor_objs = []
        
        # Ensure YCB assets are available
        if not os.path.exists(ASSET_DIR / "assets/mani_skill2_ycb"):
            download_asset.download(assets.DATA_SOURCES["ycb"])
        
        # Select YCB objects as distractors
        ycb_distractors = [
            "002_master_chef_can",
            "003_cracker_box",
            "004_sugar_box",
            "005_tomato_soup_can",
            "006_mustard_bottle",
            "007_tuna_fish_can",
            "008_pudding_box",
            "009_gelatin_box",
            "010_potted_meat_can",
            "021_bleach_cleanser",
            "024_bowl",
            "025_mug",
            "035_power_drill",
            "036_wood_block",
            "037_scissors",
            "040_large_marker",
            "051_large_clamp",
            "052_extra_large_clamp",
            "061_foam_brick",
        ]
        
        num_distractors = random.randint(1, 2)
        selected_distractors = random.sample(ycb_distractors, num_distractors)
        
        with torch.device(self.device):
            b = len(env_idx)
            
            # Get positions of the four cubes for StackCube environment
            cube_positions = []
            for cube in [self.cubeA, self.cubeB, self.cubeC, self.cubeD]:
                cube_pos = cube.pose.p[0] if cube.pose.p.dim() > 1 else cube.pose.p
                cube_positions.append(cube_pos[:2])  # Only x, y positions
            
            # Define candidate positions for distractors
            candidate_positions = [
                [-0.25, 0.25],   # Top-left corner
                [-0.25, -0.25],  # Bottom-left corner
                [0.25, 0.25],    # Top-right corner
                [0.25, -0.25],   # Bottom-right corner
                [0.0, 0.3],      # Top edge center
                [0.0, -0.3],     # Bottom edge center
                [-0.3, 0.0],     # Left edge center
                [0.3, 0.0],      # Right edge center
                [-0.2, 0.2],     # Additional positions
                [0.2, -0.2],
            ]
            
            # Filter safe positions based on distance to all cubes
            min_safe_distance = 0.15  # Minimum distance from cubes
            safe_positions = []
            
            for pos in candidate_positions:
                pos_tensor = torch.tensor(pos, device=self.device)
                
                # Check distance to all cubes
                is_safe = True
                for cube_pos in cube_positions:
                    cube_dist = torch.linalg.norm(pos_tensor - cube_pos)
                    if cube_dist < min_safe_distance:
                        is_safe = False
                        break
                
                if is_safe:
                    safe_positions.append(pos)
            
            # If not enough safe positions, use fallback positions farther away
            if len(safe_positions) < num_distractors:
                fallback_positions = [
                    [-0.35, 0.0],
                    [0.35, 0.0],
                    [0.0, -0.35],
                    [0.0, 0.35],
                ]
                for pos in fallback_positions:
                    if pos not in safe_positions:
                        pos_tensor = torch.tensor(pos, device=self.device)
                        is_safe = True
                        for cube_pos in cube_positions:
                            cube_dist = torch.linalg.norm(pos_tensor - cube_pos)
                            if cube_dist < min_safe_distance:
                                is_safe = False
                                break
                        if is_safe:
                            safe_positions.append(pos)
            
            # Shuffle safe positions
            random.shuffle(safe_positions)
            
            # Place each distractor
            for i, selected_distractor in enumerate(selected_distractors):
                if i >= len(safe_positions):
                    print(f"Warning: Not enough safe positions for all distractors")
                    break
                    
                # Create YCB object
                builder = actors.get_actor_builder(self.scene, id=f"ycb:{selected_distractor}")
                
                # Generate unique name
                episode_id = random.randint(10000, 99999)
                unique_name = f"distractor_{selected_distractor}_{episode_id}"
                
                # Set initial position in the air
                builder.initial_pose = sapien.Pose(p=[0, 0, 1.0])
                distractor_obj = builder.build(name=unique_name)
                
                # Set physical properties
                if distractor_obj.has_collision_shapes:
                    for shape in distractor_obj._bodies[0].get_collision_shapes():
                        material = shape.get_physical_material()
                        material.static_friction = 0.8
                        material.dynamic_friction = 0.7
                        material.restitution = 0.1
                
                # Use validated safe position
                selected_position = safe_positions[i]
                
                # Set distractor object position
                distractor_xyz = torch.zeros((b, 3))
                distractor_xyz[..., 0] = selected_position[0]
                distractor_xyz[..., 1] = selected_position[1]
                distractor_xyz[..., 2] = self.cube_half_size[2] + 0.05  # Note: cube_half_size is a tensor now
                
                # Random rotation
                random_angle = random.uniform(0, 2 * np.pi)
                quat = torch.zeros((b, 4))
                quat[..., 2] = np.sin(random_angle / 2)
                quat[..., 3] = np.cos(random_angle / 2)
                
                # Set pose and velocities
                distractor_obj.set_pose(Pose.create_from_pq(p=distractor_xyz, q=quat))
                distractor_obj.set_linear_velocity(torch.zeros((b, 3)))
                distractor_obj.set_angular_velocity(torch.zeros((b, 3)))
                
                # Add to list
                self.distractor_objs.append(distractor_obj)
        
        return self.distractor_objs

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
        self.cubeD = actors.build_cube(
            self.scene, half_size=0.02, color=[0.5, 0, 0.5, 1], name="cubeD", initial_pose=sapien.Pose(p=[0, 1, 0.2])
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            xyz = torch.zeros((b, 3), device=self.device)
            xyz[:, 2] = 0.02
            
            # Fixed positions for the four cubes
            cubeA_xy = torch.zeros((b, 2), device=self.device)
            cubeA_xy[:, 0] = -0.05  # fixed x position for cubeA (red)
            cubeA_xy[:, 1] = -0.05  # fixed y position for cubeA
            
            cubeB_xy = torch.zeros((b, 2), device=self.device)
            cubeB_xy[:, 0] = 0.05   # fixed x position for cubeB (green)
            cubeB_xy[:, 1] = -0.05  # fixed y position for cubeB
            
            cubeC_xy = torch.zeros((b, 2), device=self.device)
            cubeC_xy[:, 0] = 0.0    # fixed x position for cubeC (blue)
            cubeC_xy[:, 1] = 0.05   # fixed y position for cubeC
            
            cubeD_xy = torch.zeros((b, 2), device=self.device)
            cubeD_xy[:, 0] = -0.1   # fixed x position for cubeD (purple)
            cubeD_xy[:, 1] = 0.0    # fixed y position for cubeD

            # Fixed orientation (no rotation randomization)
            fixed_q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(b, 1)

            # Cube A (red)
            xyz[:, :2] = cubeA_xy
            self.cubeA.set_pose(Pose.create_from_pq(p=xyz.clone(), q=fixed_q))

            # Cube B (green)
            xyz[:, :2] = cubeB_xy
            self.cubeB.set_pose(Pose.create_from_pq(p=xyz.clone(), q=fixed_q))
            
            # Cube C (blue)
            xyz[:, :2] = cubeC_xy
            self.cubeC.set_pose(Pose.create_from_pq(p=xyz.clone(), q=fixed_q))
            
            # Cube D (purple)
            xyz[:, :2] = cubeD_xy
            self.cubeD.set_pose(Pose.create_from_pq(p=xyz, q=fixed_q))

            self._add_ycb_distractor_to_pushcube(env_idx)

    def evaluate(self):
        pos_A = self.cubeA.pose.p
        pos_B = self.cubeB.pose.p
        pos_C = self.cubeC.pose.p
        pos_D = self.cubeD.pose.p

        offset_AB = pos_A - pos_B
        offset_BC = pos_B - pos_C
        offset_AC = pos_A - pos_C
        offset_DA = pos_D - pos_A
        offset_DB = pos_D - pos_B

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

        # Original success conditions
        success_A_B = evaluate_cube_distance(offset_AB, self.cubeA, self.cubeB, "next_to")  # red next to green
        success_C_B = evaluate_cube_distance(offset_BC, self.cubeC, self.cubeB, "top")     # blue on top of green
        success_C_A = evaluate_cube_distance(offset_AC, self.cubeC, self.cubeA, "top")     # blue on top of red
        
        # New success conditions for purple cube
        success_D_A = evaluate_cube_distance(offset_DA, self.cubeD, self.cubeA, "next_to") # purple next to red
        success_D_B = evaluate_cube_distance(offset_DB, self.cubeD, self.cubeB, "next_to") # purple next to green
        
        # Purple cube should be next to either red or green cube
        success_D = torch.logical_or(success_D_A, success_D_B)
        
        # Overall success: original conditions + purple cube placement
        success = torch.logical_and(
            torch.logical_and(success_A_B, torch.logical_and(success_C_B, success_C_A)),
            success_D
        )
        
        return {
            "success": success,
            "success_A_B": success_A_B,
            "success_C_B": success_C_B,
            "success_C_A": success_C_A,
            "success_D": success_D,
        }

    def _get_obs_extra(self, info: Dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if "state" in self.obs_mode:
            obs.update(
                cubeA_pose=self.cubeA.pose.raw_pose,
                cubeB_pose=self.cubeB.pose.raw_pose,
                cubeC_pose=self.cubeC.pose.raw_pose,
                cubeD_pose=self.cubeD.pose.raw_pose,
                tcp_to_cubeA_pos=self.cubeA.pose.p - self.agent.tcp.pose.p,
                tcp_to_cubeB_pos=self.cubeB.pose.p - self.agent.tcp.pose.p,
                tcp_to_cubeC_pos=self.cubeC.pose.p - self.agent.tcp.pose.p,
                tcp_to_cubeD_pos=self.cubeD.pose.p - self.agent.tcp.pose.p,
                cubeA_to_cubeB_pos=self.cubeB.pose.p - self.cubeA.pose.p,
                cubeB_to_cubeC_pos=self.cubeC.pose.p - self.cubeB.pose.p,
                cubeA_to_cubeC_pos=self.cubeC.pose.p - self.cubeA.pose.p,
                cubeD_to_cubeA_pos=self.cubeA.pose.p - self.cubeD.pose.p,
                cubeD_to_cubeB_pos=self.cubeB.pose.p - self.cubeD.pose.p,
                cubeD_to_cubeC_pos=self.cubeC.pose.p - self.cubeD.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Dict, action: torch.Tensor, info: Dict):

        reward = torch.zeros(self.num_envs, device=self.device)
        
        # Get cube positions
        pos_A = self.cubeA.pose.p  # Red cube
        pos_B = self.cubeB.pose.p  # Green cube
        pos_C = self.cubeC.pose.p  # Blue cube
        pos_D = self.cubeD.pose.p  # Purple cube
        tcp_pos = self.agent.tcp.pose.p
        
        # Calculate distances between cubes
        dist_AB = torch.linalg.norm(pos_A - pos_B, axis=1)
        dist_AC = torch.linalg.norm(pos_A - pos_C, axis=1)
        dist_BC = torch.linalg.norm(pos_B - pos_C, axis=1)
        dist_DA = torch.linalg.norm(pos_D - pos_A, axis=1)
        dist_DB = torch.linalg.norm(pos_D - pos_B, axis=1)
        
        # Target distances
        target_next_to_dist = torch.linalg.norm(2 * self.cube_half_size[:2])  # Side by side
        target_stack_xy_dist = torch.linalg.norm(2 * self.cube_half_size[:2])  # Same XY for stacking
        target_stack_z_height = 2 * self.cube_half_size[2]  # One cube height above
        
        # 1. Reaching rewards - encourage approaching cubes that need to be moved
        tcp_to_cubes = {
            'A': torch.linalg.norm(tcp_pos - pos_A, axis=1),
            'B': torch.linalg.norm(tcp_pos - pos_B, axis=1),
            'C': torch.linalg.norm(tcp_pos - pos_C, axis=1),
            'D': torch.linalg.norm(tcp_pos - pos_D, axis=1)
        }
        
        # Progressive reaching rewards based on task sequence
        # First: position red and green cubes next to each other
        if not info.get("success_A_B", torch.tensor(False)).any():
            reaching_reward = (1.0 - torch.tanh(3.0 * tcp_to_cubes['A'])) * 0.5
            reaching_reward += (1.0 - torch.tanh(3.0 * tcp_to_cubes['B'])) * 0.5
            reward += reaching_reward
        # Then: stack blue cube on top
        elif not (info.get("success_C_A", torch.tensor(False)) & info.get("success_C_B", torch.tensor(False))).any():
            reaching_reward = (1.0 - torch.tanh(3.0 * tcp_to_cubes['C'])) * 1.0
            reward += reaching_reward
        # Finally: position purple cube next to arrangement
        else:
            reaching_reward = (1.0 - torch.tanh(3.0 * tcp_to_cubes['D'])) * 1.0
            reward += reaching_reward
        
        # 2. Grasping rewards
        grasping_rewards = {
            'A': self.agent.is_grasping(self.cubeA).float() * 1.0,
            'B': self.agent.is_grasping(self.cubeB).float() * 1.0,
            'C': self.agent.is_grasping(self.cubeC).float() * 1.0,
            'D': self.agent.is_grasping(self.cubeD).float() * 1.0
        }
        
        # Give grasping reward based on current task priority
        any_grasped = sum(grasping_rewards.values()) > 0
        reward += min(sum(grasping_rewards.values()), 1.0) * 1.5  # Cap total grasping reward
        
        # 3. Positioning rewards for red-green adjacency
        xy_dist_AB = torch.linalg.norm(pos_A[:, :2] - pos_B[:, :2], axis=1)
        red_green_adjacency_reward = 1.0 - torch.tanh(5.0 * (xy_dist_AB - target_next_to_dist))
        red_green_adjacency_reward = torch.clamp(red_green_adjacency_reward, 0, 1) * 2.0
        
        # Height alignment for red and green cubes (should be at same level)
        height_diff_AB = torch.abs(pos_A[:, 2] - pos_B[:, 2])
        height_alignment_AB = (1.0 - torch.tanh(10.0 * height_diff_AB)) * 0.5
        
        reward += red_green_adjacency_reward + height_alignment_AB
        
        # 4. Stacking rewards for blue cube on red and green
        # Blue cube should be above both red and green cubes
        blue_above_red = pos_C[:, 2] > pos_A[:, 2] + 0.01  # Small threshold
        blue_above_green = pos_C[:, 2] > pos_B[:, 2] + 0.01
        
        # XY alignment with red cube
        xy_dist_AC = torch.linalg.norm(pos_A[:, :2] - pos_C[:, :2], axis=1)
        blue_red_alignment = 1.0 - torch.tanh(5.0 * xy_dist_AC)
        
        # XY alignment with green cube  
        xy_dist_BC = torch.linalg.norm(pos_B[:, :2] - pos_C[:, :2], axis=1)
        blue_green_alignment = 1.0 - torch.tanh(5.0 * xy_dist_BC)
        
        # Height positioning (blue should be exactly one cube height above)
        z_offset_AC = pos_C[:, 2] - pos_A[:, 2]
        z_offset_BC = pos_C[:, 2] - pos_B[:, 2]
        target_z_offset = 2 * self.cube_half_size[2]
        
        height_reward_AC = 1.0 - torch.tanh(10.0 * torch.abs(z_offset_AC - target_z_offset))
        height_reward_BC = 1.0 - torch.tanh(10.0 * torch.abs(z_offset_BC - target_z_offset))
        
        # Combine stacking rewards
        stacking_reward = torch.zeros_like(reward)
        valid_stack_positions = blue_above_red & blue_above_green
        stacking_reward[valid_stack_positions] = (
            (blue_red_alignment + blue_green_alignment + height_reward_AC + height_reward_BC)[valid_stack_positions] * 0.75
        )
        reward += stacking_reward
        
        # 5. Purple cube positioning (next to the arrangement)
        # Purple should be close to either red or green cube
        purple_red_adjacency = 1.0 - torch.tanh(3.0 * (dist_DA - target_next_to_dist * 1.5))
        purple_green_adjacency = 1.0 - torch.tanh(3.0 * (dist_DB - target_next_to_dist * 1.5))
        purple_positioning = torch.max(purple_red_adjacency, purple_green_adjacency) * 1.5
        purple_positioning = torch.clamp(purple_positioning, 0, 1.5)
        
        # Only give purple positioning reward after basic stack is formed
        stack_partially_formed = info.get("success_A_B", torch.tensor(False))
        reward[stack_partially_formed] += purple_positioning[stack_partially_formed]
        
        # 6. Stability rewards
        stability_rewards = torch.zeros_like(reward)
        
        cubes = [self.cubeA, self.cubeB, self.cubeC, self.cubeD]
        for cube in cubes:
            is_static = cube.is_static(lin_thresh=1e-2, ang_thresh=0.5)
            is_not_grasped = ~self.agent.is_grasping(cube)
            cube_stable = is_static & is_not_grasped
            stability_rewards += cube_stable.float() * 0.2
        
        reward += stability_rewards
        
        # 7. Task completion bonuses
        completion_bonus = torch.zeros_like(reward)
        
        # Bonus for red-green adjacency
        if "success_A_B" in info:
            completion_bonus += info["success_A_B"].float() * 3.0
        
        # Bonus for blue stacking on red
        if "success_C_A" in info:
            completion_bonus += info["success_C_A"].float() * 3.0
            
        # Bonus for blue stacking on green
        if "success_C_B" in info:
            completion_bonus += info["success_C_B"].float() * 3.0
        
        # Bonus for purple positioning
        if "success_D" in info:
            completion_bonus += info["success_D"].float() * 2.0
        
        # Large bonus for overall success
        if "success" in info:
            completion_bonus += info["success"].float() * 10.0
        
        reward += completion_bonus
        
        # 8. Penalty for undesirable behaviors
        penalty = torch.zeros_like(reward)
        
        # Penalize cubes falling off table
        for cube in cubes:
            cube_pos = cube.pose.p
            too_low = cube_pos[:, 2] < 0.005  # Below table surface
            too_far = torch.linalg.norm(cube_pos[:, :2], axis=1) > 0.5  # Too far from center
            penalty[too_low | too_far] += -1.0
        
        # Penalize grasping multiple cubes simultaneously
        total_grasped = sum(grasping_rewards.values())
        penalty[total_grasped > 1] += -0.5
        
        reward += penalty
        
        # 9. Progress-based rewards (encourage steady improvement)
        if hasattr(self, '_prev_task_metrics'):
            # Reward improvement in key metrics
            prev_dist_AB, prev_stack_quality = self._prev_task_metrics
            current_stack_quality = (blue_red_alignment + blue_green_alignment).mean()
            
            # Progress in red-green positioning
            ab_progress = (prev_dist_AB - xy_dist_AB) * 5.0
            ab_progress = torch.clamp(ab_progress, -0.2, 0.5)
            
            # Progress in stacking quality
            stack_progress = (current_stack_quality - prev_stack_quality) * 2.0
            stack_progress = torch.clamp(stack_progress, -0.2, 0.5)
            
            reward += ab_progress + stack_progress
        
        # Store current metrics for next iteration
        self._prev_task_metrics = (xy_dist_AB.detach().clone(), 
                                (blue_red_alignment + blue_green_alignment).mean().detach().clone())
        
        return reward

    def compute_normalized_dense_reward(self, obs: Dict, action: torch.Tensor, info: Dict):
        dense_reward = self.compute_dense_reward(obs, action, info)
        max_reward = 32.0
        return torch.clamp(dense_reward / max_reward, 0.0, 1.0)