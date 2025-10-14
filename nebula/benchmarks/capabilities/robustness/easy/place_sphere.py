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

# Control-PlaceSphere-Medium

@register_env("Robust-PlaceSphere-Easy", max_episode_steps=50)
class RobustPlaceSphereEasyEnv(BaseEnv):
    """
    **Task Description:**
    Place the sphere into one of the two shallow bins (red or blue).

    **Randomizations:**
    - The position of the bins and the sphere are randomized: The bins are initialized in [0, 0.1] x [-0.08, 0.08],
    and the sphere is initialized in [-0.1, -0.05] x [-0.1, 0.1]

    **Success Conditions:**
    - The sphere is placed on the top of either bin. The robot remains static and the gripper is not closed at the end state.
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

    def _build_bin(self, radius, color):
        """Build a bin with specified color"""
        builder = self.scene.create_actor_builder()

        # Create material with color
        material = sapien.render.RenderMaterial()
        material.set_base_color(color)

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
            builder.add_box_visual(pose, half_size, material=material)

        # build the kinematic bin
        return builder.build_kinematic(name=f"{color}_bin")
    
    def _add_ycb_distractor_to_pushcube(self, env_idx: torch.Tensor):
        """
        Add 1-2 YCB distractor objects to the PlaceSphere environment
        
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
            
            # Get current positions of sphere and bins for PlaceSphere environment
            sphere_pos = self.obj.pose.p[0] if self.obj.pose.p.dim() > 1 else self.obj.pose.p
            red_bin_pos = self.red_bin.pose.p[0] if self.red_bin.pose.p.dim() > 1 else self.red_bin.pose.p
            blue_bin_pos = self.blue_bin.pose.p[0] if self.blue_bin.pose.p.dim() > 1 else self.blue_bin.pose.p
            
            # Define candidate positions for distractor placement
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
            
            # Filter safe positions based on distance to sphere and bins
            min_safe_distance = 0.15  # Minimum distance from sphere and bins
            safe_positions = []
            
            for pos in candidate_positions:
                pos_tensor = torch.tensor(pos, device=self.device)
                
                # Check distance to sphere
                sphere_dist = torch.linalg.norm(pos_tensor - sphere_pos[:2])
                # Check distance to red bin
                red_bin_dist = torch.linalg.norm(pos_tensor - red_bin_pos[:2])
                # Check distance to blue bin
                blue_bin_dist = torch.linalg.norm(pos_tensor - blue_bin_pos[:2])
                
                # Ensure minimum distance from sphere and both bins
                if sphere_dist > min_safe_distance and red_bin_dist > min_safe_distance and blue_bin_dist > min_safe_distance:
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
                        sphere_dist = torch.linalg.norm(pos_tensor - sphere_pos[:2])
                        red_bin_dist = torch.linalg.norm(pos_tensor - red_bin_pos[:2])
                        blue_bin_dist = torch.linalg.norm(pos_tensor - blue_bin_pos[:2])
                        if sphere_dist > min_safe_distance and red_bin_dist > min_safe_distance and blue_bin_dist > min_safe_distance:
                            safe_positions.append(pos)
            
            # Shuffle safe positions for randomness
            random.shuffle(safe_positions)
            
            # Place each distractor object
            for i, selected_distractor in enumerate(selected_distractors):
                if i >= len(safe_positions):
                    print(f"Warning: Not enough safe positions for all distractors")
                    break
                    
                # Create YCB object builder
                builder = actors.get_actor_builder(self.scene, id=f"ycb:{selected_distractor}")
                
                # Generate unique name to avoid naming conflicts
                episode_id = random.randint(10000, 99999)
                unique_name = f"distractor_{selected_distractor}_{episode_id}"
                
                # Set initial position high to avoid collision
                builder.initial_pose = sapien.Pose(p=[0, 0, 1.0])
                distractor_obj = builder.build(name=unique_name)
                
                # Set physical properties for stability
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
                distractor_xyz[..., 2] = self.radius + 0.05  # Height based on sphere radius
                
                # Apply random rotation around z-axis
                random_angle = random.uniform(0, 2 * np.pi)
                quat = torch.zeros((b, 4))
                quat[..., 2] = np.sin(random_angle / 2)
                quat[..., 3] = np.cos(random_angle / 2)
                
                # Set pose and zero velocities for stability
                distractor_obj.set_pose(Pose.create_from_pq(p=distractor_xyz, q=quat))
                distractor_obj.set_linear_velocity(torch.zeros((b, 3)))
                distractor_obj.set_angular_velocity(torch.zeros((b, 3)))
                
                # Add to tracking list
                self.distractor_objs.append(distractor_obj)
        
        return self.distractor_objs

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        # load the table
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # load the sphere
        self.obj = actors.build_sphere(
            self.scene,
            radius=self.radius,
            color=np.array([12, 42, 160, 255]) / 255,
            name="sphere",
            body_type="dynamic",
            initial_pose=Pose.create_from_pq(p=[0, 0, self.radius], q=[1, 0, 0, 0])
        )

        # load the two bins - red and blue
        self.red_bin = self._build_bin(self.radius, color=np.array([0.50196078, 0.0, 0.50196078, 1.0]))
        
        self.blue_bin = self._build_bin(self.radius, color=np.array([0.0, 0.0, 1.0, 1.0]))
        
        # Keep reference to bins for compatibility
        self.bins = [self.red_bin, self.blue_bin]

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            # init the table scene
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # init the sphere at a fixed position in the first 1/4 zone along the x-axis
            xyz = torch.zeros((b, 3))
            xyz[..., 0] = -0.075  # fixed x position in first 1/4 zone (middle of [-0.1, -0.05])
            xyz[..., 1] = 0.0     # fixed y position (center)
            xyz[..., 2] = self.radius  # on the table
            q = [1, 0, 0, 0]
            obj_pose = Pose.create_from_pq(p=xyz, q=q)
            self.obj.set_pose(obj_pose)

            # init the two bins on the same line along the x-axis
            # Red bin positioned at y = -0.05, Blue bin at y = 0.05
            
            # Red bin position
            red_pos = torch.zeros((b, 3))
            red_pos[:, 0] = 0.05  # fixed x position
            red_pos[:, 1] = -0.05  # fixed y position for red bin
            red_pos[:, 2] = self.block_half_size[0]  # on the table
            red_bin_pose = Pose.create_from_pq(p=red_pos, q=[1, 0, 0, 0])
            self.red_bin.set_pose(red_bin_pose)

            # Blue bin position  
            blue_pos = torch.zeros((b, 3))
            blue_pos[:, 0] = 0.05  # same fixed x position as red bin
            blue_pos[:, 1] = 0.05  # fixed y position for blue bin
            blue_pos[:, 2] = self.block_half_size[0]  # on the table
            blue_bin_pose = Pose.create_from_pq(p=blue_pos, q=[1, 0, 0, 0])
            self.blue_bin.set_pose(blue_bin_pose)

            self._add_ycb_distractor_to_pushcube(env_idx)

            self.task_phase = torch.zeros(len(env_idx), dtype=torch.int32, device=self.device)  # 0: initial, 1: first placed, 2: completed
            self.has_placed_in_first_bin = torch.zeros(len(env_idx), dtype=torch.bool, device=self.device)
            self.has_placed_in_second_bin = torch.zeros(len(env_idx), dtype=torch.bool, device=self.device)

    def evaluate(self):
        pos_obj = self.obj.pose.p
        
        is_obj_on_purple_bin = self._check_obj_on_bin(pos_obj, self.red_bin.pose.p) 
        is_obj_on_blue_bin = self._check_obj_on_bin(pos_obj, self.blue_bin.pose.p)
        is_obj_grasped = self.agent.is_grasping(self.obj)
        is_robot_static = self.agent.is_static(0.2)
        
        first_placement_success = (self.task_phase == 0) & is_obj_on_purple_bin & (~is_obj_grasped) & is_robot_static
        self.has_placed_in_first_bin = torch.where(
            first_placement_success,
            torch.ones_like(self.has_placed_in_first_bin),
            self.has_placed_in_first_bin
        )
        self.task_phase = torch.where(
            first_placement_success,
            torch.ones_like(self.task_phase),
            self.task_phase
        )
        
        second_placement_success = (self.task_phase == 1) & is_obj_on_blue_bin & (~is_obj_grasped) & is_robot_static
        self.has_placed_in_second_bin = torch.where(
            second_placement_success,
            torch.ones_like(self.has_placed_in_second_bin),
            self.has_placed_in_second_bin
        )
        self.task_phase = torch.where(
            second_placement_success,
            torch.full_like(self.task_phase, 2),
            self.task_phase
        )
        
        success = self.has_placed_in_first_bin & self.has_placed_in_second_bin
        
        return {
            "is_obj_grasped": is_obj_grasped,
            "is_obj_on_purple_bin": is_obj_on_purple_bin,
            "is_obj_on_blue_bin": is_obj_on_blue_bin,
            "is_robot_static": is_robot_static,
            "task_phase": self.task_phase,
            "has_placed_in_first_bin": self.has_placed_in_first_bin,
            "has_placed_in_second_bin": self.has_placed_in_second_bin,
            "success": success,
        }

    def _check_obj_on_bin(self, pos_obj, pos_bin):
        """Check if object is on top of a specific bin"""
        offset = pos_obj - pos_bin
        xy_flag = torch.linalg.norm(offset[..., :2], axis=1) <= 0.005
        z_flag = (
            torch.abs(offset[..., 2] - self.radius - self.block_half_size[0]) <= 0.005
        )
        return torch.logical_and(xy_flag, z_flag)

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            is_grasped=info["is_obj_grasped"],
            tcp_pose=self.agent.tcp.pose.raw_pose,
            red_bin_pos=self.red_bin.pose.p,
            blue_bin_pos=self.blue_bin.pose.p,
        )
        if "state" in self.obs_mode:
            obs.update(
                obj_pose=self.obj.pose.raw_pose,
                tcp_to_obj_pos=self.obj.pose.p - self.agent.tcp.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_pose = self.agent.tcp.pose.p
        obj_pos = self.obj.pose.p
        
        obj_to_tcp_dist = torch.linalg.norm(tcp_pose - obj_pos, axis=1)
        reaching_reward = 2 * (1 - torch.tanh(5 * obj_to_tcp_dist))
        
        reward = reaching_reward.clone()
        
        purple_bin_top_pos = self.red_bin.pose.p.clone()  # red_bin实际是紫色
        purple_bin_top_pos[:, 2] += self.block_half_size[0] + self.radius
        obj_to_purple_dist = torch.linalg.norm(purple_bin_top_pos - obj_pos, axis=1)
        
        phase_0_mask = info["task_phase"] == 0
        if phase_0_mask.any():
            purple_placement_reward = 1 - torch.tanh(5.0 * obj_to_purple_dist)
            reward[phase_0_mask & info["is_obj_grasped"]] = (4 + purple_placement_reward)[phase_0_mask & info["is_obj_grasped"]]
            
            reward[phase_0_mask & info["is_obj_on_purple_bin"] & (~info["is_obj_grasped"])] = 8
        
        blue_bin_top_pos = self.blue_bin.pose.p.clone()
        blue_bin_top_pos[:, 2] += self.block_half_size[0] + self.radius
        obj_to_blue_dist = torch.linalg.norm(blue_bin_top_pos - obj_pos, axis=1)
        
        phase_1_mask = info["task_phase"] == 1
        if phase_1_mask.any():
            blue_placement_reward = 1 - torch.tanh(5.0 * obj_to_blue_dist)
            reward[phase_1_mask & info["is_obj_grasped"]] = (4 + blue_placement_reward)[phase_1_mask & info["is_obj_grasped"]]
            reward[phase_1_mask & info["is_obj_on_blue_bin"] & (~info["is_obj_grasped"])] = 10
        
        reward[info["success"]] = 15
        
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
        # this should be equal to compute_dense_reward / max possible reward
        max_reward = 13.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward