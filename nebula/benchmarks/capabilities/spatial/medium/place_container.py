from typing import Any, Dict, Union, List
import numpy as np
import sapien
import torch
import random
import os

import nebula.core.simulation.utils.randomization as randomization
from nebula.core.embodiment.robots import Fetch, Panda
from nebula.core.simulation.engine import BaseEnv
from nebula.core.sensors.camera import CameraConfig
from nebula.utils import sapien_utils
from nebula.utils.building import actors
from nebula.utils.registration import register_env
from nebula.utils.scene_builder.table import TableSceneBuilder
from nebula.utils.structs.pose import Pose

SPATIAL_PLACE_CONTAINER_DOC_STRING = """Task Description:
Place a movable cube in specified 3D spatial positions relative to a container.
The robot must understand 3D spatial relationships including inside/beside and above/below
concepts, demonstrating comprehension of volumetric space and vertical positioning.

3D Spatial Relations:
- Inside: The cube should be placed within the container's interior volume
- Beside: The cube should be placed external to the container, nearby but not inside
- Above: The cube should be placed above the container at a higher Z coordinate
- Below: The cube should be placed below the container at a lower Z coordinate

Task Components:
- One container (YCB bowl) suspended in the air as 3D reference
- One movable cube (red) that needs to be positioned
- Clear 3D spatial instruction: "Place the red cube [inside/beside/above/below] the container"
- Success requires accurate 3D spatial positioning relative to the container

Container Positioning:
- Container is positioned above the table surface to enable "below" positioning
- Container height allows for clear inside/beside/above/below distinctions
- Container size is appropriate for cube manipulation tasks

Randomizations:
- Container position is randomized in 3D space (with constraints)
- Target spatial relationship (inside/beside/above/below) is randomized
- Initial cube position is randomized on the table surface
- Container orientation can be slightly randomized

Success Conditions:
- Cube is placed in the correct 3D spatial region relative to container
- Cube maintains stable position after placement
- Spatial relationship is clearly satisfied (inside bounds, beside bounds, etc.)
- Placement accuracy is within acceptable 3D tolerance
"""

@register_env("SpatialReasoning-PlaceContainer-Medium", max_episode_steps=120)
class SpatialReasoningMediumPlaceContainerEnv(BaseEnv):
    
    SUPPORTED_ROBOTS = ["panda", "fetch"]
    agent: Union[Panda, Fetch]
    
    cube_half_size = 0.02
    container_radius = 0.06  # Default container radius
    container_height = 0.04  # Default container height
    
    # 3D Spatial configuration
    CONTAINER_HEIGHT = 0.15      # Height above table for container
    INSIDE_TOLERANCE = 0.02      # Tolerance for being "inside"
    BESIDE_MIN_DISTANCE = 0.08   # Minimum distance for "beside"
    VERTICAL_OFFSET = 0.08       # Offset for "top" and "bottom"
    PLACEMENT_TOLERANCE = 0.03   # General placement tolerance
    
    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        
        # 3D spatial directions
        self.spatial_directions_3d = ["inside", "beside", "top", "bottom"]
        
        # Color definitions
        self.color_rgbs = {
            "red": [1, 0, 0, 1],        # Movable cube color
            "light_blue": [0.7, 0.8, 1.0, 1],  # Light container color for better visibility
            "gray": [0.5, 0.5, 0.5, 1]  # Alternative container color
        }
        
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
        pose = sapien_utils.look_at(eye=[0.6, 0.6, 0.8], target=[0, 0, 0.15])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.table_scene.build()
        
        # Create movable cube (red)
        self.movable_cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=self.color_rgbs["red"],
            name="movable_cube",
            initial_pose=sapien.Pose(p=[0, 0, 1.0])
        )
        
        from nebula import ASSET_DIR
        if not os.path.exists(ASSET_DIR / "assets/mani_skill2_ycb"):
            from nebula.utils import assets, download_asset
            download_asset.download(assets.DATA_SOURCES["ycb"])

        builder = actors.get_actor_builder(
            self.scene, 
            id="ycb:024_bowl"
        )
        builder.set_physx_body_type("kinematic")
        builder.initial_pose = sapien.Pose(p=[0, 0, 1.0])
        self.container = builder.build(name="container")

        self.container_radius = 0.08
        self.container_height = 0.035

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            
            # Choose target 3D spatial direction
            self.target_3d_direction = random.choice(self.spatial_directions_3d)
            
            # Generate instruction
            direction_phrases = {
                "inside": "inside",
                "beside": "beside",
                "top": "above",
                "bottom": "below"
            }
          
            self.task_instruction = f"Place the red cube {direction_phrases[self.target_3d_direction]} the metal bowl"
            
            # Position container in 3D space
            self._position_container(b)
            
            # Position movable cube on table
            self._position_movable_cube(b)
            
            # Calculate target 3D position
            self._calculate_target_3d_position()
            
            # Let physics settle
            for _ in range(10):
                self.scene.step()

    def _position_container(self, b):
        """Position the container at an elevated position in 3D space"""
        container_xyz = torch.zeros((b, 3))
        # Random XY position with some constraints
        container_xyz[:, :2] = (torch.rand((b, 2)) * 2 - 1) * 0.08  # Small area around center
        # Fixed height above table
        container_xyz[:, 2] = self.CONTAINER_HEIGHT
        
        self.container.set_pose(Pose.create_from_pq(container_xyz))
        self.container_pos = container_xyz[0].clone()
        
        # print(f"Container position: ({self.container_pos[0]:.3f}, {self.container_pos[1]:.3f}, {self.container_pos[2]:.3f})")

    def _position_movable_cube(self, b):
        """Position the movable cube on the table surface"""
        cube_xyz = torch.zeros((b, 3))
        # Random position on table, away from container projection
        angle = torch.rand(1) * 2 * np.pi
        distance = 0.12 + torch.rand(1) * 0.08  # 0.12-0.20m from center
        cube_xyz[:, 0] = distance * torch.cos(angle)
        cube_xyz[:, 1] = distance * torch.sin(angle)
        cube_xyz[:, 2] = self.cube_half_size  # On table surface
        
        self.movable_cube.set_pose(Pose.create_from_pq(cube_xyz))
        self.initial_cube_pos = cube_xyz[0].clone()

    def _calculate_target_3d_position(self):
        """Calculate the target 3D position based on spatial instruction"""
        container_center = self.container_pos.clone()
        
        if self.target_3d_direction == "inside":
            # Target position is inside the container (same XY, slightly lower Z)
            self.target_3d_pos = container_center.clone()
            self.target_3d_pos[2] = container_center[2] - self.container_height + self.cube_half_size
            
        elif self.target_3d_direction == "beside":
            # Target position is beside the container (nearby but outside radius)
            angle = torch.rand(1) * 2 * np.pi
            offset_distance = self.container_radius + self.BESIDE_MIN_DISTANCE
            self.target_3d_pos = container_center.clone()
            self.target_3d_pos[0] += offset_distance * torch.cos(angle).item()
            self.target_3d_pos[1] += offset_distance * torch.sin(angle).item()
            self.target_3d_pos[2] = container_center[2] - self.container_height + self.cube_half_size
            
        elif self.target_3d_direction == "top":
            # Target position is above the container
            self.target_3d_pos = container_center.clone()
            self.target_3d_pos[2] = container_center[2] + self.container_height + self.VERTICAL_OFFSET
            
        elif self.target_3d_direction == "bottom":
            # Target position is below the container (on table, under container)
            self.target_3d_pos = container_center.clone()
            self.target_3d_pos[2] = self.cube_half_size  # On table surface
        
        # print(f"Target 3D position: ({self.target_3d_pos[0]:.3f}, {self.target_3d_pos[1]:.3f}, {self.target_3d_pos[2]:.3f})")

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            target_3d_direction_encoded=self._encode_3d_direction(self.target_3d_direction),
            task_type_encoded=self._encode_task_type("place_container_3d"),
        )
        
        if "state" in self.obs_mode:
            obs.update(
                container_pose=self.container.pose.raw_pose,
                movable_cube_pose=self.movable_cube.pose.raw_pose,
                target_3d_position=self.target_3d_pos,
                container_center=self.container_pos,
                container_radius=torch.tensor(self.container_radius, device=self.device),
                container_height=torch.tensor(self.container_height, device=self.device),
                tcp_to_cube_pos=self.movable_cube.pose.p - self.agent.tcp.pose.p,
                tcp_to_container_pos=self.container.pose.p - self.agent.tcp.pose.p,
                cube_to_container_pos=self.container.pose.p - self.movable_cube.pose.p,
                cube_to_target_pos=self.target_3d_pos - self.movable_cube.pose.p,
                container_to_target_pos=self.target_3d_pos - self.container_pos,
                is_grasping_cube=self.agent.is_grasping(self.movable_cube),
                cube_height_above_table=self.movable_cube.pose.p[:, 2] - self.cube_half_size,
                container_height_above_table=self.container_pos[2] - self.container_height,
            )
            
        return obs

    def _encode_3d_direction(self, direction):
        """Encode 3D direction as integer: inside=0, beside=1, top=2, bottom=3"""
        direction_map = {"inside": 0, "beside": 1, "top": 2, "bottom": 3}
        return direction_map.get(direction, 0)

    def _encode_task_type(self, task_type):
        """Encode task type as integer: place_container_3d=0, place_between=1, pick_closest=2"""
        task_map = {"place_container_3d": 0, "place_between": 1, "pick_closest": 2}
        return task_map.get(task_type, 0)

    def evaluate(self):
        """Evaluate if the cube is correctly placed in the target 3D spatial region"""
        cube_pos = self.movable_cube.pose.p[0]
        
        # Check spatial relationship based on target direction
        correct_3d_placement = self._verify_3d_spatial_relationship(cube_pos)
        
        # Check if cube is stable (not moving)
        cube_velocity = torch.linalg.norm(self.movable_cube.linear_velocity[0])
        is_stable = cube_velocity < 0.02
        
        # Check if cube hasn't fallen through the world
        cube_above_ground = cube_pos[2] > 0.01
        
        # Check if robot is not interfering
        robot_clear = not self.agent.is_grasping(self.movable_cube)
        
        # Additional check: cube should be within reasonable bounds
        within_workspace = (torch.abs(cube_pos[0]) < 0.5) and (torch.abs(cube_pos[1]) < 0.5) and (cube_pos[2] < 0.5)
        
        success = (correct_3d_placement & is_stable & cube_above_ground & 
                  robot_clear & within_workspace)
        
        return {
            "success": success,
            "correct_3d_placement": correct_3d_placement,
            "is_stable": is_stable,
            "cube_above_ground": cube_above_ground,
            "robot_clear": robot_clear,
            "within_workspace": within_workspace,
            "task_instruction": self.task_instruction,
            "target_3d_direction": self.target_3d_direction,
            "cube_position": cube_pos
        }

    def _verify_3d_spatial_relationship(self, cube_pos):
        """Verify that the cube is in the correct 3D spatial relationship to container"""
        container_center = self.container_pos
        
        if self.target_3d_direction == "inside":
            # Check if cube is inside container bounds
            xy_distance = torch.linalg.norm(cube_pos[:2] - container_center[:2])
            inside_radius = xy_distance < (self.container_radius - self.INSIDE_TOLERANCE)
            
            z_inside = (cube_pos[2] > (container_center[2] - self.container_height)) and \
                      (cube_pos[2] < (container_center[2] + self.container_height))
            
            return inside_radius and z_inside
            
        elif self.target_3d_direction == "beside":
            # Check if cube is beside container - outside radius but at similar height
            xy_distance = torch.linalg.norm(cube_pos[:2] - container_center[:2])
            beside_radius = xy_distance > self.container_radius + self.PLACEMENT_TOLERANCE
            
            # Should be at approximately the same level as container bottom
            z_level = torch.abs(cube_pos[2] - (container_center[2] - self.container_height + self.cube_half_size)) < self.PLACEMENT_TOLERANCE
            
            return beside_radius and z_level
            
        elif self.target_3d_direction == "top":
            # Check if cube is above the container
            xy_distance = torch.linalg.norm(cube_pos[:2] - container_center[:2])
            above_container = xy_distance < (self.container_radius + self.PLACEMENT_TOLERANCE)
            
            z_above = cube_pos[2] > (container_center[2] + self.container_height + self.PLACEMENT_TOLERANCE)
            
            return above_container and z_above
            
        elif self.target_3d_direction == "bottom":
            # Check if cube is below the container (on table, under container projection)
            xy_distance = torch.linalg.norm(cube_pos[:2] - container_center[:2])
            below_container = xy_distance < (self.container_radius + self.PLACEMENT_TOLERANCE)
            
            z_below = cube_pos[2] < (container_center[2] - self.container_height - self.PLACEMENT_TOLERANCE)
            on_table = torch.abs(cube_pos[2] - self.cube_half_size) < self.PLACEMENT_TOLERANCE
            
            return below_container and z_below and on_table
        
        return torch.tensor(False)

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        cube_pos = self.movable_cube.pose.p[0]
        
        # 1. Reaching reward - encourage robot to approach the cube first
        tcp_to_cube_dist = torch.linalg.norm(
            self.movable_cube.pose.p - self.agent.tcp_pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_cube_dist)
        
        # 2. Grasping reward
        grasping_reward = self.agent.is_grasping(self.movable_cube).float() * 2
        
        # 3. 3D Navigation reward - encourage moving toward target 3D area
        cube_to_target_dist = torch.linalg.norm(cube_pos - self.target_3d_pos)
        navigation_3d_reward = (1 - torch.tanh(3 * cube_to_target_dist)) * 3
        
        # 4. Direction-specific rewards
        direction_reward = self._compute_direction_specific_reward(cube_pos)
        
        # 5. Height achievement reward (for top/bottom tasks)
        height_reward = self._compute_height_reward(cube_pos)
        
        # 6. Stability reward
        cube_velocity = torch.linalg.norm(self.movable_cube.linear_velocity[0])
        stability_reward = torch.exp(-cube_velocity * 5) * 0.5
        
        # 7. Workspace constraint reward
        workspace_reward = self._compute_workspace_reward(cube_pos)
        
        reward = (reaching_reward + grasping_reward + navigation_3d_reward + 
                 direction_reward + height_reward + stability_reward + workspace_reward)
        
        # Success bonus
        if info["success"]:
            reward += 5
            
        return reward

    def _compute_direction_specific_reward(self, cube_pos):
        """Reward specific to the target 3D direction"""
        container_center = self.container_pos
        
        if self.target_3d_direction == "inside":
            # Reward for being close to container center in XY
            xy_distance = torch.linalg.norm(cube_pos[:2] - container_center[:2])
            xy_reward = (1 - torch.tanh(5 * xy_distance / self.container_radius)) * 1.5
            return xy_reward
            
        elif self.target_3d_direction == "beside":
            # Reward for being at appropriate distance beside container
            xy_distance = torch.linalg.norm(cube_pos[:2] - container_center[:2])
            target_distance = self.container_radius + self.BESIDE_MIN_DISTANCE
            distance_diff = torch.abs(xy_distance - target_distance)
            beside_reward = (1 - torch.tanh(5 * distance_diff)) * 1.5
            return beside_reward
            
        elif self.target_3d_direction == "top":
            # Reward for being above container
            if cube_pos[2] > container_center[2]:
                height_diff = cube_pos[2] - container_center[2]
                top_reward = torch.clamp(height_diff * 10, 0, 1.5)
                return top_reward
            else:
                return torch.tensor(0.0)
                
        elif self.target_3d_direction == "bottom":
            # Reward for being below container but above ground
            if cube_pos[2] < container_center[2] and cube_pos[2] > 0.01:
                bottom_reward = 1.5
                return bottom_reward
            else:
                return torch.tensor(0.0)
        
        return torch.tensor(0.0)

    def _compute_height_reward(self, cube_pos):
        """Reward for achieving correct height based on target direction"""
        if self.target_3d_direction == "top":
            # Encourage lifting the cube high
            lift_reward = torch.clamp(cube_pos[2] * 5, 0, 2)
            return lift_reward
        elif self.target_3d_direction == "inside":
            # Encourage appropriate height for inside container
            target_height = self.container_pos[2] - self.container_height + self.cube_half_size
            height_diff = torch.abs(cube_pos[2] - target_height)
            height_reward = (1 - torch.tanh(10 * height_diff)) * 1
            return height_reward
        else:
            # For beside and bottom, prefer table level
            table_height = self.cube_half_size
            height_diff = torch.abs(cube_pos[2] - table_height)
            height_reward = (1 - torch.tanh(10 * height_diff)) * 0.5
            return height_reward

    def _compute_workspace_reward(self, cube_pos):
        """Reward for keeping cube within reasonable workspace"""
        # Penalize if cube goes too far from working area
        distance_from_origin = torch.linalg.norm(cube_pos[:2])
        if distance_from_origin > 0.3:
            return torch.tensor(-1.0)
        else:
            return torch.tensor(0.0)

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 15
    
    def get_task_instruction(self):
        return self.task_instruction

SpatialReasoningMediumPlaceContainerEnv.__doc__ = SPATIAL_PLACE_CONTAINER_DOC_STRING