from typing import Any, Dict, Union, List
import numpy as np
import sapien
import torch
import random

import nebula.core.simulation.utils.randomization as randomization
from nebula.core.embodiment.robots import Fetch, Panda
from nebula.core.simulation.engine import BaseEnv
from nebula.core.sensors.camera import CameraConfig
from nebula.utils import sapien_utils
from nebula.utils.building import actors
from nebula.utils.registration import register_env
from nebula.utils.scene_builder.table import TableSceneBuilder
from nebula.utils.structs.pose import Pose

SPATIAL_EASY_DOC_STRING = """Task Description:
Pick the specified colored cube based on spatial reference to the blue landmark cube. 
Each spatial direction (left/right/front/back) contains exactly one colored cube, eliminating ambiguity.

Spatial Relations:
- Left/Right: Based on robot's perspective with clear separation
- Front/Back: Based on robot's perspective with clear separation
- Reference object is always a blue cube at the center
- Each direction contains exactly ONE colored cube

Unambiguous Design:
- Spatial exclusivity: Each direction (left/right/front/back) has only one colored cube
- Clear separation: Minimum distance between cubes ensures distinct spatial regions
- No overlapping zones: Each cube belongs to exactly one spatial direction
- Precise instructions: "Pick the red cube to the left of the blue one" has only one valid target

Randomizations:
- Reference cube (blue) position is randomized on the table
- Color-to-direction assignments are randomized each episode
- Small positional variations within each spatial region

Success Conditions:
- Pick the correct colored cube that satisfies the spatial relation
- Cube is lifted above minimum height (0.05m)
- Robot is static after completion
"""

@register_env("SpatialReasoning-PickCube-Easy", max_episode_steps=75)
class SpatialReasoningEasyPickCubeEnv(BaseEnv):
    
    SUPPORTED_ROBOTS = ["panda", "fetch"]
    agent: Union[Panda, Fetch]
    
    cube_half_size = 0.02
    lift_thresh = 0.05
    
    # Spatial configuration for unambiguous positioning
    SPATIAL_SEPARATION = 0.12      # Minimum distance between cubes in different directions
    REGION_VARIANCE = 0.03         # Small random variation within each spatial region
    REFERENCE_AREA_SIZE = 0.08     # Size of central area for reference cube
    
    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        
        # Spatial directions
        self.spatial_directions = ["left", "right", "front", "back"]
        
        # Available colors (excluding blue which is the reference)
        self.available_colors = ["red", "green", "yellow", "purple"]
        
        # Color definitions
        self.cube_colors = {
            "red": [1, 0, 0, 1],
            "green": [0, 1, 0, 1], 
            "blue": [0, 0, 1, 1],    # Always the reference cube
            "yellow": [1, 1, 0, 1],
            "purple": [1, 0, 1, 1]
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
        pose = sapien_utils.look_at(eye=[0.6, 0.6, 0.8], target=[0, 0, 0.1])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.table_scene.build()
        
        # Create cubes for each color
        self.cubes = {}
        for color_name, color_rgb in self.cube_colors.items():
            if color_name != "blue":
                cube = actors.build_cube(
                    self.scene,
                    half_size=self.cube_half_size,
                    color=color_rgb,
                    name=f"{color_name}_cube",
                    initial_pose=sapien.Pose(p=[0, 0, 1.0])  # Start high to avoid collision
                )
            else:
                # Blue cube is the reference cube
                cube = actors.build_cube(
                    self.scene,
                    half_size=self.cube_half_size,
                    color=color_rgb,  
                    name=f"{color_name}_cube",
                    body_type = "static",
                    add_collision=False,
                    initial_pose=sapien.Pose(p=[0, 0, 1.0])
                )
            self.cubes[color_name] = cube

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            
            # Create unambiguous spatial assignment
            # Each direction gets exactly one color, randomly assigned
            colors_for_assignment = random.sample(self.available_colors, 4)
            self.spatial_assignments = {
                direction: color 
                for direction, color in zip(self.spatial_directions, colors_for_assignment)
            }
            
            # Choose target direction and corresponding color
            self.target_direction = random.choice(self.spatial_directions)
            self.target_color = self.spatial_assignments[self.target_direction]
            
            # Generate unambiguous instruction
            self.task_instruction = f"Pick the cube to the {self.target_direction} of the blue one"
            
            #print(f"Spatial Instruction: '{self.task_instruction}'")
            #print(f"Spatial Assignments: {self.spatial_assignments}")
            #print(f"Target: {self.target_color} cube ({self.target_direction})")
            
            # Position reference cube (blue) in center area
            self._position_reference_cube(b)
            
            # Position colored cubes in their assigned spatial directions
            self._position_spatial_cubes(b)
            
            # Let physics settle
            for _ in range(10):
                self.scene.step()

    def _position_reference_cube(self, b):
        """Position the blue reference cube in the center area"""
        ref_xyz = torch.zeros((b, 3))
        # Place in center area with small random variation
        ref_xyz[:, :2] = (torch.rand((b, 2)) * 2 - 1) * self.REFERENCE_AREA_SIZE
        ref_xyz[:, 2] = self.cube_half_size
        self.cubes["blue"].set_pose(Pose.create_from_pq(ref_xyz))
        
        # Store reference position for spatial calculations
        self.reference_pos = ref_xyz[0, :2].clone()

    def _position_spatial_cubes(self, b):
        """Position colored cubes in their assigned spatial directions with clear separation"""
        
        # Define spatial offset vectors from reference cube
        spatial_offsets = {
            "left": torch.tensor([0.0, self.SPATIAL_SEPARATION]),      # Positive Y is left from robot perspective
            "right": torch.tensor([0.0, -self.SPATIAL_SEPARATION]),    # Negative Y is right from robot perspective  
            "front": torch.tensor([self.SPATIAL_SEPARATION, 0.0]),     # Positive X is front (closer to robot)
            "back": torch.tensor([-self.SPATIAL_SEPARATION, 0.0])      # Negative X is back (away from robot)
        }
        
        # Position each assigned cube in its spatial direction
        for direction, color in self.spatial_assignments.items():
            cube_xyz = torch.zeros((b, 3))
            
            # Base position = reference position + spatial offset
            base_pos = self.reference_pos + spatial_offsets[direction]
            
            # Add small random variation within the spatial region
            variation = (torch.rand((b, 2)) * 2 - 1) * self.REGION_VARIANCE
            cube_xyz[:, :2] = base_pos + variation
            cube_xyz[:, 2] = self.cube_half_size
            
            self.cubes[color].set_pose(Pose.create_from_pq(cube_xyz))
            
            # Store cube position for verification
            setattr(self, f"{direction}_cube_pos", cube_xyz[0, :2].clone())

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            target_direction_encoded=self._encode_direction(self.target_direction),
            target_color_encoded=self._encode_color(self.target_color),
        )
        
        if "state" in self.obs_mode:
            # Add reference cube information
            reference_cube = self.cubes["blue"]
            target_cube = self.cubes[self.target_color]
            
            obs.update(
                reference_cube_pose=reference_cube.pose.raw_pose,
                target_cube_pose=target_cube.pose.raw_pose,
                tcp_to_target_pos=target_cube.pose.p - self.agent.tcp.pose.p,
                tcp_to_reference_pos=reference_cube.pose.p - self.agent.tcp.pose.p,
                target_to_reference_pos=reference_cube.pose.p - target_cube.pose.p,
            )
            
            # Add all cube poses
            for color_name, cube in self.cubes.items():
                obs[f"{color_name}_cube_pose"] = cube.pose.raw_pose
                obs[f"is_grasping_{color_name}"] = self.agent.is_grasping(cube)
        
        return obs

    def _encode_direction(self, direction):
        """Encode direction as integer: left=0, right=1, front=2, back=3"""
        direction_map = {"left": 0, "right": 1, "front": 2, "back": 3}
        return direction_map.get(direction, 0)

    def _encode_color(self, color):
        """Encode color as integer: red=0, green=1, yellow=2, purple=3, blue=4"""
        color_map = {"red": 0, "green": 1, "yellow": 2, "purple": 3, "blue": 4}
        return color_map.get(color, 0)

    def evaluate(self):
        """Evaluate if the correct cube (based on spatial instruction) is picked"""
        target_cube = self.cubes[self.target_color]
        
        # Check basic task completion
        is_lifted = target_cube.pose.p[:, 2] > self.cube_half_size + self.lift_thresh
        is_grasped = self.agent.is_grasping(target_cube)
        is_robot_static = self.agent.is_static(0.2)
        
        # Check if wrong cube was picked (spatial understanding failure)
        wrong_cube_picked = False
        for color, cube in self.cubes.items():
            if color != self.target_color and color != "blue":  # Exclude reference cube
                if self.agent.is_grasping(cube):
                    wrong_cube_picked = True
                    break
        
        # Verify spatial relationship is maintained
        spatial_relationship_correct = self._verify_spatial_relationship()
        
        success = (is_lifted & is_grasped & is_robot_static & 
                  (~wrong_cube_picked) & spatial_relationship_correct)
        
        return {
            "success": success,
            "is_lifted": is_lifted,
            "is_grasped": is_grasped,
            "is_robot_static": is_robot_static,
            "wrong_cube_picked": wrong_cube_picked,
            "spatial_relationship_correct": spatial_relationship_correct,
            "task_instruction": self.task_instruction,
            "target_color": self.target_color,
            "target_direction": self.target_direction
        }

    def _verify_spatial_relationship(self):
        """Verify that the target cube is still in the correct spatial relationship to reference"""
        target_cube = self.cubes[self.target_color]
        reference_cube = self.cubes["blue"]
        
        target_pos = target_cube.pose.p[0, :2]
        ref_pos = reference_cube.pose.p[0, :2]
        
        relative_pos = target_pos - ref_pos
        
        # Check if cube is in the correct spatial direction
        if self.target_direction == "left":
            return relative_pos[1] > (self.SPATIAL_SEPARATION * 0.5)  # Positive Y
        elif self.target_direction == "right":
            return relative_pos[1] < -(self.SPATIAL_SEPARATION * 0.5)  # Negative Y
        elif self.target_direction == "front":
            return relative_pos[0] > (self.SPATIAL_SEPARATION * 0.5)  # Positive X
        elif self.target_direction == "back":
            return relative_pos[0] < -(self.SPATIAL_SEPARATION * 0.5)  # Negative X
        
        return torch.tensor(True)

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        target_cube = self.cubes[self.target_color]
        
        # 1. Reaching reward for correct target
        tcp_to_target_dist = torch.linalg.norm(
            target_cube.pose.p - self.agent.tcp_pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_target_dist)
        
        # 2. Grasping reward for correct target
        grasping_reward = self.agent.is_grasping(target_cube).float() * 2
        
        # 3. Lifting reward for correct target
        lift_height = target_cube.pose.p[:, 2] - self.cube_half_size
        lifting_reward = torch.clamp(lift_height / self.lift_thresh, 0, 1) * 2
        
        # 4. Spatial understanding bonus
        spatial_bonus = self._compute_spatial_understanding_bonus()
        
        # 5. Wrong cube penalty (emphasizes spatial discrimination)
        wrong_cube_penalty = info["wrong_cube_picked"] * 3.0
        
        # 6. Spatial relationship maintenance bonus
        spatial_correct_bonus = info["spatial_relationship_correct"].float() * 1.0
        
        reward = (reaching_reward + grasping_reward + lifting_reward + 
                 spatial_bonus + spatial_correct_bonus - wrong_cube_penalty)
        
        # Success bonus
        reward[info["success"]] += 3
            
        return reward

    def _compute_spatial_understanding_bonus(self):
        """Bonus for exploring the correct spatial region"""
        target_cube = self.cubes[self.target_color]
        reference_cube = self.cubes["blue"]
        
        # Bonus for being in the correct spatial quadrant
        tcp_pos = self.agent.tcp_pose.p[0, :2]
        ref_pos = reference_cube.pose.p[0, :2]
        target_pos = target_cube.pose.p[0, :2]
        
        # Check if TCP is moving toward the correct spatial direction
        tcp_to_ref = tcp_pos - ref_pos
        target_to_ref = target_pos - ref_pos
        
        # Dot product to measure alignment with correct direction
        alignment = torch.sum(tcp_to_ref * target_to_ref) / (
            torch.linalg.norm(tcp_to_ref) * torch.linalg.norm(target_to_ref) + 1e-6
        )
        
        spatial_exploration_bonus = torch.clamp(alignment, 0, 1) * 0.5
        
        # Additional bonus for being close to target area
        tcp_to_target_dist = torch.linalg.norm(tcp_pos - target_pos)
        proximity_bonus = (1 - torch.tanh(3 * tcp_to_target_dist)) * 0.3
        
        return spatial_exploration_bonus + proximity_bonus

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 9
    
    def get_task_instruction(self):
        return self.task_instruction

SpatialReasoningEasyPickCubeEnv.__doc__ = SPATIAL_EASY_DOC_STRING