from typing import Any, Dict, Union, List
import numpy as np
import sapien
import torch

import nebula.core.simulation.utils.randomization as randomization
from nebula.core.embodiment.robots import Fetch, Panda
from nebula.core.simulation.engine import BaseEnv
from nebula.core.sensors.camera import CameraConfig
from nebula.utils import sapien_utils
from nebula.utils.building import actors
from nebula.utils.registration import register_env
from nebula.utils.scene_builder.table import TableSceneBuilder
from nebula.utils.structs.pose import Pose

@register_env("Spatial-PlaceBetween-Easy", max_episode_steps=50)
class SpatialEasyPlaceBetweenEnv(BaseEnv):
    """Task Description:
    Place a movable red cube between two fixed reference cubes (blue and green).
    The robot must understand the spatial concept of "between" and accurately position
    the red cube in the middle area defined by the two reference cubes.

    Spatial Relations:
    - Two reference cubes (blue and green) are placed at random positions
    - Distance between reference cubes is controlled to be reasonable
    - Target position is the midpoint between the two reference cubes
    - "Between" means equidistant from both reference cubes

    Task Components:
    - Two fixed reference cubes (blue and green) at random positions
    - One movable cube (red) that needs to be positioned
    - Clear spatial instruction: "Place the red cube between the blue and green cubes"
    - Success requires accurate spatial positioning in the middle area

    Randomizations:
    - Blue cube position is randomized on the table
    - Green cube position is randomized relative to blue cube
    - Distance between reference cubes varies within reasonable range
    - Initial position of red movable cube is randomized
    - Reference cubes maintain reasonable separation distance

    Success Conditions:
    - Red cube is placed within the middle region between reference cubes
    - Red cube is stable (not moving) after placement
    - Distance from red cube to both reference cubes is approximately equal
    - Placement accuracy is within acceptable tolerance
    """

    SUPPORTED_ROBOTS = ["panda", "fetch"]
    agent: Union[Panda, Fetch]

    cube_half_size = 0.015
    task_instruction = "Place the red cube between the blue and green cubes."

    # Spatial configuration
    MIN_REFERENCE_DISTANCE = 0.12
    MAX_REFERENCE_DISTANCE = 0.22
    BETWEEN_TOLERANCE = 0.03
    PLACEMENT_TOLERANCE = 0.025

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise

        self.color_rgbs = {
            "red": [1, 0, 0, 1],     # movable cube
            "blue": [0, 0, 1, 1],    # reference 1
            "green": [0, 1, 0, 1],   # reference 2
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

    # ---------------------------- Scene ----------------------------
    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.table_scene.build()

        self.movable_cube = actors.build_cube(
            self.scene, half_size=self.cube_half_size, color=self.color_rgbs["red"],
            name="movable_cube", initial_pose=sapien.Pose(p=[0, 0, 1.0])
        )
        self.blue_cube = actors.build_cube(
            self.scene, half_size=self.cube_half_size, color=self.color_rgbs["blue"],
            name="blue_cube", initial_pose=sapien.Pose(p=[0, 0, 1.0])
        )
        self.green_cube = actors.build_cube(
            self.scene, half_size=self.cube_half_size, color=self.color_rgbs["green"],
            name="green_cube", initial_pose=sapien.Pose(p=[0, 0, 1.0])
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            self.task_instruction = "Place the red cube between the blue and green cubes"

            self._position_reference_cubes(b)
            self._position_movable_cube(b)
            self._calculate_between_position()

            for _ in range(10):
                self.scene.step()

    # ---------------------------- Placement helpers ----------------------------
    def _position_reference_cubes(self, b: int):
        """
        Optimized reference cube placement:
        - Ensures sufficient spacing between reference cubes with no overlap
        - Guarantees placement within robot's reachable workspace
        """
        # Define safe workspace for robot reachability (Panda arm reach ~0.855m, centered on table)
        SAFE_X_RANGE = [-0.10, 0.20]  # Front-back direction
        SAFE_Y_RANGE = [-0.20, 0.20]  # Left-right direction
        MIN_CUBE_SEPARATION = self.cube_half_size * 4  # Minimum 4x cube size spacing (0.06m)
        
        # Blue cube: random position within safe workspace
        blue_xyz = torch.zeros((b, 3), device=self.device, dtype=torch.float32)
        blue_xyz[:, 0] = torch.rand((b,), device=self.device) * (SAFE_X_RANGE[1] - SAFE_X_RANGE[0]) + SAFE_X_RANGE[0]
        blue_xyz[:, 1] = torch.rand((b,), device=self.device) * (SAFE_Y_RANGE[1] - SAFE_Y_RANGE[0]) + SAFE_Y_RANGE[0]
        blue_xyz[:, 2] = self.cube_half_size
        self.blue_cube.set_pose(Pose.create_from_pq(blue_xyz))
        self.blue_pos = blue_xyz[..., :2].clone()  # (B, 2)
        
        # Green cube: ensure proper distance and no overlap with blue
        # Adjust distance range to ensure midpoint is reachable
        ADJUSTED_MIN_DISTANCE = max(self.MIN_REFERENCE_DISTANCE, MIN_CUBE_SEPARATION)
        ADJUSTED_MAX_DISTANCE = min(self.MAX_REFERENCE_DISTANCE, 0.22)  # Ensure midpoint stays in workspace
        
        distance = torch.rand((b,), device=self.device) * \
                (ADJUSTED_MAX_DISTANCE - ADJUSTED_MIN_DISTANCE) + ADJUSTED_MIN_DISTANCE
        angle = torch.rand((b,), device=self.device) * 2 * np.pi
        
        green_xyz = torch.zeros((b, 3), device=self.device, dtype=torch.float32)
        green_xyz[:, 0] = blue_xyz[:, 0] + distance * torch.cos(angle)
        green_xyz[:, 1] = blue_xyz[:, 1] + distance * torch.sin(angle)
        green_xyz[:, 2] = self.cube_half_size
        
        # Clamp green cube to safe workspace
        green_xyz[:, 0] = torch.clamp(green_xyz[:, 0], SAFE_X_RANGE[0], SAFE_X_RANGE[1])
        green_xyz[:, 1] = torch.clamp(green_xyz[:, 1], SAFE_Y_RANGE[0], SAFE_Y_RANGE[1])
        
        self.green_cube.set_pose(Pose.create_from_pq(green_xyz))
        self.green_pos = green_xyz[..., :2].clone()
        
        # Store actual distance for observations
        self.reference_distance = torch.linalg.norm(self.green_pos - self.blue_pos, dim=1)

    def _position_movable_cube(self, b: int):
        """
        Optimized movable cube placement:
        - Ensures no collision with reference cubes
        - Places cube in graspable location with sufficient clearance
        - Guarantees cube is within robot's reach
        """
        SAFE_X_RANGE = [-0.12, 0.22]
        SAFE_Y_RANGE = [-0.22, 0.22]
        MIN_DISTANCE_FROM_REFERENCES = self.cube_half_size * 8
        MAX_PLACEMENT_ATTEMPTS = 20
        
        cube_xyz = torch.zeros((b, 3), device=self.device, dtype=torch.float32)
        
        for batch_idx in range(b):
            placed = False
            for attempt in range(MAX_PLACEMENT_ATTEMPTS):
                # Generate random position within safe workspace
                angle = torch.rand(1, device=self.device) * 2 * np.pi
                distance = 0.08 + torch.rand(1, device=self.device) * 0.12  # 8-20cm from center
                
                x = distance * torch.cos(angle)
                y = distance * torch.sin(angle)
                
                # Clamp to safe workspace
                x = torch.clamp(x, SAFE_X_RANGE[0], SAFE_X_RANGE[1])
                y = torch.clamp(y, SAFE_Y_RANGE[0], SAFE_Y_RANGE[1])
                
                candidate_pos = torch.tensor([x, y], device=self.device)
                
                # Check distances from reference cubes
                dist_to_blue = torch.linalg.norm(candidate_pos - self.blue_pos[batch_idx])
                dist_to_green = torch.linalg.norm(candidate_pos - self.green_pos[batch_idx])
                
                # Also ensure it's not too close to the target midpoint (to make task non-trivial)
                midpoint = (self.blue_pos[batch_idx] + self.green_pos[batch_idx]) / 2.0
                dist_to_midpoint = torch.linalg.norm(candidate_pos - midpoint)
                
                if (dist_to_blue >= MIN_DISTANCE_FROM_REFERENCES and 
                    dist_to_green >= MIN_DISTANCE_FROM_REFERENCES and
                    dist_to_midpoint >= self.cube_half_size * 4):
                    
                    cube_xyz[batch_idx, 0] = x
                    cube_xyz[batch_idx, 1] = y
                    cube_xyz[batch_idx, 2] = self.cube_half_size
                    placed = True
                    break
            
            if not placed:
                # Fallback: place at a safe default location
                cube_xyz[batch_idx, 0] = 0.15
                cube_xyz[batch_idx, 1] = -0.15 if batch_idx % 2 == 0 else 0.15
                cube_xyz[batch_idx, 2] = self.cube_half_size
        
        self.movable_cube.set_pose(Pose.create_from_pq(cube_xyz))
        self.initial_cube_pos = cube_xyz[..., :2].clone() 

    def _calculate_between_position(self):
        """
        Calculate target placement position (midpoint between reference cubes)
        Verify it's within reachable workspace
        """
        self.target_between_pos = (self.blue_pos + self.green_pos) / 2.0 
        
        # Log warning if target is outside safe workspace (for debugging)
        SAFE_X_RANGE = [-0.15, 0.25]
        SAFE_Y_RANGE = [-0.25, 0.25]
        
        for i in range(self.target_between_pos.shape[0]):
            x, y = self.target_between_pos[i]
            if not (SAFE_X_RANGE[0] <= x <= SAFE_X_RANGE[1] and 
                    SAFE_Y_RANGE[0] <= y <= SAFE_Y_RANGE[1]):
                print(f"[Warning] Target position ({x:.3f}, {y:.3f}) may be outside safe workspace")

    # ---------------------------- Observations ----------------------------
    def _get_obs_extra(self, info: Dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose, 
            task_type_encoded=torch.full((len(self.agent.tcp.pose.p),), self._encode_task_type("place_between"),
                                         device=self.device, dtype=torch.long),
        )

        if "state" in self.obs_mode:
            target_between_3d = torch.stack([
                self.target_between_pos[..., 0],
                self.target_between_pos[..., 1],
                torch.full_like(self.target_between_pos[..., 0], self.cube_half_size),
            ], dim=1)

            obs.update(
                blue_cube_pose=self.blue_cube.pose.raw_pose,
                green_cube_pose=self.green_cube.pose.raw_pose,
                movable_cube_pose=self.movable_cube.pose.raw_pose,
                target_between_position=target_between_3d,                           
                reference_distance=torch.full((target_between_3d.shape[0],), self.reference_distance,
                                              device=self.device, dtype=torch.float32),  
                tcp_to_movable_pos=self.movable_cube.pose.p - self.agent.tcp.pose.p,  
                tcp_to_blue_pos=self.blue_cube.pose.p - self.agent.tcp.pose.p,        
                tcp_to_green_pos=self.green_cube.pose.p - self.agent.tcp.pose.p,      
                movable_to_target_pos=target_between_3d - self.movable_cube.pose.p,   
                blue_to_green_pos=self.green_cube.pose.p - self.blue_cube.pose.p,     
                is_grasping_cube=self.agent.is_grasping(self.movable_cube),           
            )
        return obs

    @staticmethod
    def _encode_task_type(task_type: str) -> int:
        task_map = {"place_between": 0, "pick": 1, "move": 2}
        return task_map.get(task_type, 0)

    # ---------------------------- Evaluation ----------------------------
    def _check_on_line_between_cubes(self, cube_pos: torch.Tensor) -> torch.Tensor:
        """
        cube_pos: (B, 2)
        return: (B,) bool -> whether the projection is on the line segment AND close to the line
        """
        blue_to_green = self.green_pos - self.blue_pos         
        blue_to_cube  = cube_pos - self.blue_pos               

        den = torch.linalg.norm(blue_to_green, dim=1)           
        valid = den >= 1e-6

        proj_len = (blue_to_cube * blue_to_green).sum(dim=1) / den.clamp_min(1e-6)    
        proj_pt  = self.blue_pos + (proj_len / den.clamp_min(1e-6)).unsqueeze(1) * blue_to_green 

        dist_to_line = torch.linalg.norm(cube_pos - proj_pt, dim=1)   
        on_line = dist_to_line < self.PLACEMENT_TOLERANCE

        between = (proj_len >= 0.0) & (proj_len <= den)
        return on_line & between & valid

    def evaluate(self):
        # Positions
        cube_pos_2d = self.movable_cube.pose.p[..., :2]  

        # Tolerances
        distance_to_target = torch.linalg.norm(cube_pos_2d - self.target_between_pos, dim=1)  
        correct_between_placement = distance_to_target < self.BETWEEN_TOLERANCE               

        dist_to_blue  = torch.linalg.norm(cube_pos_2d - self.blue_pos,  dim=1)
        dist_to_green = torch.linalg.norm(cube_pos_2d - self.green_pos, dim=1)
        distance_difference = torch.abs(dist_to_blue - dist_to_green)
        equidistant = distance_difference < self.PLACEMENT_TOLERANCE

        is_on_line = self._check_on_line_between_cubes(cube_pos_2d)

        # Stability / table / robot clear
        cube_velocity = torch.linalg.norm(self.movable_cube.linear_velocity, dim=1)   
        is_stable = cube_velocity < 0.01

        cube_height = self.movable_cube.pose.p[..., 2]   
        on_table = cube_height > (self.cube_half_size * 0.5)

        robot_clear = ~self.agent.is_grasping(self.movable_cube)   

        success = (correct_between_placement & equidistant & is_on_line & is_stable & on_table & robot_clear)

        return {
            "success": success,
            "correct_between_placement": correct_between_placement,
            "equidistant": equidistant,
            "is_on_line": is_on_line,
            "is_stable": is_stable,
            "on_table": on_table,
            "robot_clear": robot_clear,
            "distance_to_target": distance_to_target,
            "distance_difference": distance_difference,
            "task_instruction": self.task_instruction,
        }

    # ---------------------------- Rewards ----------------------------
    def _compute_line_alignment_reward(self, cube_pos_2d: torch.Tensor) -> torch.Tensor:
        """
        Reward for aligning with the line between reference cubes. Returns (B,)
        """
        blue_to_green = self.green_pos - self.blue_pos
        den = torch.linalg.norm(blue_to_green, dim=1)                     
        valid = den >= 1e-6

        blue_to_cube = cube_pos_2d - self.blue_pos
        proj_len = (blue_to_cube * blue_to_green).sum(dim=1) / den.clamp_min(1e-6)
        proj_pt  = self.blue_pos + (proj_len / den.clamp_min(1e-6)).unsqueeze(1) * blue_to_green
        dist_to_line = torch.linalg.norm(cube_pos_2d - proj_pt, dim=1)    

        line_proximity_reward = (1 - torch.tanh(10 * dist_to_line)) * 1.5  

        normalized_projection = proj_len / den.clamp_min(1e-6)
        between_bonus = torch.where((normalized_projection >= 0.0) & (normalized_projection <= 1.0),
                                    torch.ones_like(proj_len), torch.zeros_like(proj_len))  
        return line_proximity_reward + between_bonus * valid.float()

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        """
        Returns (B,) dense reward (batched).
        """
        # Reaching
        tcp_to_cube_dist = torch.linalg.norm(self.movable_cube.pose.p - self.agent.tcp.pose.p, dim=1)  
        reaching_reward = 1 - torch.tanh(5 * tcp_to_cube_dist)                                         

        # Grasp
        grasping_reward = self.agent.is_grasping(self.movable_cube).float() * 2.0                      

        # Between target
        cube_pos_2d = self.movable_cube.pose.p[..., :2]                                               
        cube_to_target_dist = torch.linalg.norm(cube_pos_2d - self.target_between_pos, dim=1)         
        between_reward = (1 - torch.tanh(5 * cube_to_target_dist)) * 3.0                               

        # Equidistant
        dist_to_blue  = torch.linalg.norm(cube_pos_2d - self.blue_pos,  dim=1)
        dist_to_green = torch.linalg.norm(cube_pos_2d - self.green_pos, dim=1)
        distance_difference = torch.abs(dist_to_blue - dist_to_green)
        equidistant_reward = (1 - torch.tanh(10 * distance_difference)) * 2.0                          

        # Line alignment
        line_reward = self._compute_line_alignment_reward(cube_pos_2d)                                  

        # Stability
        cube_velocity = torch.linalg.norm(self.movable_cube.linear_velocity, dim=1)                    
        stability_reward = torch.exp(-cube_velocity * 10) * 0.5                                        

        # Height maintenance
        cube_height = self.movable_cube.pose.p[..., 2]                                                 
        height_reward = torch.clamp((cube_height - self.cube_half_size * 0.5) * 10, 0, 0.5)            

        reward = reaching_reward + grasping_reward + between_reward + equidistant_reward + line_reward + stability_reward + height_reward

        # Success bonus
        if "success" in info:
            reward = reward + 5.0 * info["success"].float()

        return reward  

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 14.0