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

SPATIAL_HARD_BUILD_DOC_STRING = """Task Description:
Test spatial understanding by building a house-like structure with multi-step stacking.
Robot must demonstrate understanding of 3D spatial relationships and sequential construction.

Construction Sequence:
1. Place first cube (base) at designated building area
2. Stack second cube on top of first cube
3. Place triangular prism (roof) on top of the cube tower

Spatial Understanding Requirements:
- Stacking stability: Each piece must remain stable when stacked
- No collapse: Structure must not fall over during or after construction
- Sequential order: Must follow the correct building sequence (cube1 -> cube2 -> triangle)

Randomizations:
- Initial positions of all three pieces are randomized on table
- Building target area is randomized within reachable workspace
- Slight variations in piece orientations to test adaptation

Success Conditions:
- All three pieces are stacked in correct order (cube1 -> cube2 -> triangle)
- Final structure is stable (no pieces falling or moving)
- Structure remains standing without collapse
- Robot is static after completion

Challenge Aspects:
- Requires understanding of 3D spatial relationships
- Tests stability assessment rather than precision alignment
- Demands sequential task planning
- Evaluates physical intuition about stacking stability
"""

@register_env("Spatial-BuildBlock-Hard", max_episode_steps=150)
class SpatialHardBuildBlockEnv(BaseEnv):
    
    SUPPORTED_ROBOTS = ["panda", "fetch"]
    agent: Union[Panda, Fetch]
    
    # Object dimensions
    cube_half_size = 0.025
    triangle_base_size = 0.08
    triangle_height = 0.06
    
    # Task parameters
    stability_threshold = 0.005     # Maximum movement for stability check
    building_area_size = 0.15       # Size of the building area
    min_stack_height = 0.04         # Minimum height to consider "stacked"
    
    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        
        # Building sequence tracking
        self.build_sequence = ["cube1", "cube2", "triangle"]
        self.current_step = 0

        self.task_instruction = "Create a three-level tower: red cube at bottom, green cube in middle, blue triangle at top."
        
        # Object colors for visual distinction
        self.object_colors = {
            "cube1": [1, 0, 0, 1],      # Red cube (base)
            "cube2": [0, 1, 0, 1],      # Green cube (middle)
            "triangle": [0, 0, 1, 1]    # Blue triangle (roof)
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
        
        # Create building objects
        self.objects = {}
        
        # Create two cubes
        for i in range(2):
            cube_name = f"cube{i+1}"
            cube = actors.build_cube(
                self.scene,
                half_size=self.cube_half_size,
                color=self.object_colors[cube_name],
                name=cube_name,
                initial_pose=sapien.Pose(p=[0, 0, 1.0])  # Start high to avoid collision
            )
            self.objects[cube_name] = cube

        # Create triangular prism (roof)
        triangle = self._create_triangular_prism()
        self.objects["triangle"] = triangle

    def _create_triangular_prism(self):
        """Create a triangular prism with tip pointing upward (Z-axis)"""
        import trimesh
        import os
        
        bottom_vertices = np.array([
            [-self.triangle_base_size/2, 0, 0], 
            [self.triangle_base_size/2, 0, 0],  
            [0, 0, self.triangle_height]  
        ])
        
        top_vertices = bottom_vertices.copy()
        top_vertices[:, 1] = self.triangle_base_size/2
        
        vertices = np.vstack([bottom_vertices, top_vertices])
        
        faces = np.array([
            [0, 1, 2], [3, 5, 4],  
            [0, 3, 4], [0, 4, 1], 
            [1, 4, 5], [1, 5, 2], 
            [2, 5, 3], [2, 3, 0]  
        ])
    
        # Create the mesh
        prism_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        blue_color = [0, 0, 255, 255]
        prism_mesh.visual.vertex_colors = np.tile(blue_color, (len(vertices), 1))
        
        # Save to a persistent location
        mesh_dir = os.path.join(os.path.dirname(__file__), "assets")
        os.makedirs(mesh_dir, exist_ok=True)
        mesh_path = os.path.join(mesh_dir, "triangular_prism.obj")
        
        # Only create the file if it doesn't exist
        if not os.path.exists(mesh_path):
            prism_mesh.export(mesh_path)
            
            mtl_path = mesh_path.replace('.obj', '.mtl')
            with open(mtl_path, 'w') as f:
                f.write("newmtl blue_material\n")
                f.write("Ka 0.0 0.0 1.0\n")  
                f.write("Kd 0.0 0.0 1.0\n")  
                f.write("Ks 0.5 0.5 0.5\n")  
                f.write("Ns 100.0\n")   
                f.write("d 1.0\n")  
            
            with open(mesh_path, 'r') as f:
                obj_content = f.read()
            
            with open(mesh_path, 'w') as f:
                f.write(f"mtllib {os.path.basename(mtl_path)}\n")
                f.write("usemtl blue_material\n")
                f.write(obj_content)
        
        # Create actor using mesh file
        builder = self.scene.create_actor_builder()
        builder.add_convex_collision_from_file(mesh_path)
        builder.add_visual_from_file(mesh_path)
        builder.set_initial_pose(sapien.Pose(p=[0, 0, 1.0]))
        triangle_actor = builder.build(name="triangle")
        
        return triangle_actor

    def _position_initial_objects(self, b):
        """
        Safely position objects on table surface without overlaps
        """
        table_bounds = {
            'x_min': -0.15, 'x_max': 0.15,
            'y_min': -0.15, 'y_max': 0.15
        }
        
        object_radii = {
            "cube1": 0.04,     # cube_half_size + buffer
            "cube2": 0.04,     # cube_half_size + buffer  
            "triangle": 0.06   # triangle_base_size/2 + buffer
        }
        
        placed_positions = []
        
        for obj_name, obj in self.objects.items():
            obj_radius = object_radii[obj_name]
            max_attempts = 50
            
            for attempt in range(max_attempts):
                x = torch.rand(1, device=self.device) * (table_bounds['x_max'] - table_bounds['x_min']) + table_bounds['x_min']
                y = torch.rand(1, device=self.device) * (table_bounds['y_max'] - table_bounds['y_min']) + table_bounds['y_min']
                
                candidate_pos = torch.tensor([x.item(), y.item()], device=self.device)
                
                building_center_2d = self.building_center[:2]
                dist_to_center = torch.linalg.norm(candidate_pos - building_center_2d)
                
                if dist_to_center < 0.08:
                    continue
                
                valid_position = True
                for placed_pos, placed_radius in placed_positions:
                    dist_to_placed = torch.linalg.norm(candidate_pos - placed_pos)
                    min_distance = obj_radius + placed_radius + 0.02 
                    
                    if dist_to_placed < min_distance:
                        valid_position = False
                        break
                
                if valid_position:
                    placed_positions.append((candidate_pos, obj_radius))
                    
                    obj_xyz = torch.zeros((b, 3), device=self.device)
                    obj_xyz[:, :2] = candidate_pos.unsqueeze(0).expand(b, -1)
                    
                    if obj_name == "triangle":
                        obj_xyz[:, 2] = self.triangle_height/2
                        random_angle = torch.tensor([np.pi], device=self.device)
                        quat = torch.zeros((b, 4), device=self.device)
                        quat[:, 3] = torch.cos(random_angle / 2)
                        quat[:, 2] = torch.sin(random_angle / 2) 
                    else:
                        obj_xyz[:, 2] = self.cube_half_size
                        
                    quat = torch.zeros((b, 4), device=self.device)
                    quat[:, 3] = 1.0
                    
                    obj.set_pose(Pose.create_from_pq(obj_xyz, quat))
                    
                    #print(f"[DEBUG] {obj_name} placed at {candidate_pos.cpu().numpy()} after {attempt+1} attempts")
                    break
            else:
                #print(f"[WARNING] Could not find safe position for {obj_name}, using fallback")
                fallback_positions = {
                    "cube1": torch.tensor([-0.12, -0.12], device=self.device),
                    "cube2": torch.tensor([0.12, -0.12], device=self.device),
                    "triangle": torch.tensor([0.0, -0.12], device=self.device)
                }
                
                fallback_pos = fallback_positions[obj_name]
                placed_positions.append((fallback_pos, obj_radius))
                
                obj_xyz = torch.zeros((b, 3), device=self.device)
                obj_xyz[:, :2] = fallback_pos.unsqueeze(0).expand(b, -1)
                
                if obj_name == "triangle":
                    obj_xyz[:, 2] = self.triangle_height/2
                else:
                    obj_xyz[:, 2] = self.cube_half_size
                    
                quat = torch.zeros((b, 4), device=self.device)
                quat[:, 3] = 1.0
                
                obj.set_pose(Pose.create_from_pq(obj_xyz, quat))

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            
            self.current_step = 0
            self.building_center = torch.tensor([0.05, 0.0], dtype=torch.float32, device=self.device)
            random_offset = (torch.rand(2, dtype=torch.float32, device=self.device) - 0.5) * 0.06
            self.building_center = self.building_center + random_offset
            
            self.building_center[0] = torch.clamp(self.building_center[0], -0.1, 0.15)
            self.building_center[1] = torch.clamp(self.building_center[1], -0.1, 0.1)
            
            #print(f"Building Task: Stack cube1 -> cube2 -> triangle")
            #print(f"Building area center: ({self.building_center[0]:.3f}, {self.building_center[1]:.3f})")
            
            self._position_initial_objects(b)
            
            self._initialize_tracking_variables(b)
            
            for _ in range(5):
                self.scene.step()

    def _check_object_overlaps(self):
        positions = {}
        for obj_name, obj in self.objects.items():
            pos = obj.pose.p
            if torch.is_tensor(pos) and pos.dim() > 1:
                pos = pos[0]
            positions[obj_name] = pos[:2] 
        
        min_distances = {
            ("cube1", "cube2"): 0.06,
            ("cube1", "triangle"): 0.08,
            ("cube2", "triangle"): 0.08
        }
        
        overlaps = []
        for (obj1, obj2), min_dist in min_distances.items():
            if obj1 in positions and obj2 in positions:
                dist = torch.linalg.norm(positions[obj1] - positions[obj2])
                if dist < min_dist:
                    overlaps.append((obj1, obj2, dist.item(), min_dist))
        
        return len(overlaps) == 0

    def _initialize_tracking_variables(self, b):
        """Initialize variables for tracking build progress"""
        self.cube1_placed = torch.zeros(b, dtype=torch.bool, device=self.device)
        self.cube2_stacked = torch.zeros(b, dtype=torch.bool, device=self.device)
        self.triangle_placed = torch.zeros(b, dtype=torch.bool, device=self.device)
            
        # Track previous positions for stability check
        self.prev_positions = {}
        for obj_name in self.objects.keys():
            self.prev_positions[obj_name] = torch.zeros((b, 3))

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            current_build_step_encoded=torch.tensor([self.current_step], device=self.device, dtype=torch.long),
            task_type_encoded=torch.tensor([self._encode_task_type("build_block")], device=self.device, dtype=torch.long),
        )
        
        if "state" in self.obs_mode:
            obs.update(
                building_center=torch.cat([
                    self.building_center.to(self.device), 
                    torch.tensor([self.cube_half_size], device=self.device)
                ]),
                building_area_size=torch.tensor([self.building_area_size], device=self.device),
                cube_half_size=torch.tensor([self.cube_half_size], device=self.device),
                triangle_height=torch.tensor([self.triangle_height], device=self.device),
            )
            
            # Safe TCP to building center distance calculation
            tcp_pos = self.agent.tcp.pose.p
            if tcp_pos.dim() > 1:
                tcp_pos_2d = tcp_pos[0, :2]
            else:
                tcp_pos_2d = tcp_pos[:2]
                
            obs["tcp_to_building_center"] = torch.cat([
                self.building_center.to(tcp_pos_2d.device) - tcp_pos_2d, 
                torch.tensor([0.0], device=tcp_pos_2d.device)
            ])
            
            # Add all object poses and grasping status
            for obj_name, obj in self.objects.items():
                obs[f"{obj_name}_pose"] = obj.pose.raw_pose
                
                is_grasping = self.agent.is_grasping(obj)
                if torch.is_tensor(is_grasping):
                    if is_grasping.dim() == 0:
                        is_grasping = is_grasping.unsqueeze(0)
                else:
                    is_grasping = torch.tensor([is_grasping], device=self.device, dtype=torch.bool)
                obs[f"is_grasping_{obj_name}"] = is_grasping
                
                obs[f"tcp_to_{obj_name}_pos"] = obj.pose.p - self.agent.tcp.pose.p
                
                obj_pos = obj.pose.p
                if obj_pos.dim() > 1:
                    obj_pos_2d = obj_pos[0, :2]
                else:
                    obj_pos_2d = obj_pos[:2]
                    
                obs[f"{obj_name}_to_building_center_pos"] = torch.cat([
                    self.building_center.to(obj_pos_2d.device) - obj_pos_2d, 
                    torch.tensor([0.0], device=obj_pos_2d.device)
                ])
                
            # Add inter-object relationships
            cube1_pos = self.objects["cube1"].pose.p
            cube2_pos = self.objects["cube2"].pose.p
            triangle_pos = self.objects["triangle"].pose.p
            
            obs.update(
                cube1_to_cube2_pos=cube2_pos - cube1_pos,
                cube2_to_triangle_pos=triangle_pos - cube2_pos,
                cube1_to_triangle_pos=triangle_pos - cube1_pos,
            )

            obs.update(
                cube1_placed=self.cube1_placed.float(),
                cube2_stacked=self.cube2_stacked.float(),
                triangle_placed=self.triangle_placed.float(),
            )
            
            # Add height information - ensure 1D tensor
            obs.update(
                cube1_height=self.objects["cube1"].pose.p[:, 2],
                cube2_height=self.objects["cube2"].pose.p[:, 2],
                triangle_height_pos=self.objects["triangle"].pose.p[:, 2],
            )
                
        return obs

    def _encode_task_type(self, task_type):
        """Encode task type as integer: build_block=0, place_container_3d=1, pick_3d_spatial=2"""
        task_map = {"build_block": 0, "place_container_3d": 1, "pick_3d_spatial": 2}
        return task_map.get(task_type, 0)

    def evaluate(self):
        cube1_pos = self._get_object_position(self.objects["cube1"])
        cube2_pos = self._get_object_position(self.objects["cube2"])
        triangle_pos = self._get_object_position(self.objects["triangle"])
        
        correct_height_order = (cube1_pos[2] < cube2_pos[2] < triangle_pos[2])
    
        horizontal_tolerance = 0.05
        
        cube1_to_cube2_dist = torch.linalg.norm(cube1_pos[:2] - cube2_pos[:2])
        cube2_to_triangle_dist = torch.linalg.norm(cube2_pos[:2] - triangle_pos[:2])
        
        cube2_aligned = cube1_to_cube2_dist < horizontal_tolerance
        triangle_aligned = cube2_to_triangle_dist < horizontal_tolerance
        
        not_grasped = (
            ~self.agent.is_grasping(self.objects["cube1"]) &
            ~self.agent.is_grasping(self.objects["cube2"]) &
            ~self.agent.is_grasping(self.objects["triangle"])
        )
        
        success = correct_height_order & cube2_aligned & triangle_aligned & not_grasped
        
        return {
            "success": success,
            "correct_height_order": correct_height_order,
            "cube2_aligned": cube2_aligned,
            "triangle_aligned": triangle_aligned,
            "not_grasped": not_grasped,
            "task_instruction": self.task_instruction,
        }

    def _get_object_position(self, obj):
        pos = obj.pose.p
        if torch.is_tensor(pos) and pos.dim() > 1:
            pos = pos[0]
        return pos

    def _check_cube1_placement(self):
        """Check if cube1 is placed in the building area"""
        cube1 = self.objects["cube1"]
        
        # Get cube1 position and handle dimensions properly
        cube1_pos = cube1.pose.p
        if torch.is_tensor(cube1_pos):
            if cube1_pos.dim() > 1:
                cube1_pos = cube1_pos[0]  # Remove batch dimension if present
        
        # Extract 2D position (x, y) from 3D position (x, y, z)
        cube1_pos_2d = cube1_pos[:2]
        
        # Ensure building_center is also 2D and on the same device
        building_center = self.building_center
        if torch.is_tensor(building_center):
            if building_center.device != cube1_pos_2d.device:
                building_center = building_center.to(cube1_pos_2d.device)
        else:
            building_center = torch.tensor(building_center, device=cube1_pos_2d.device)
        
        # Check if in building area
        dist_to_center = torch.linalg.norm(cube1_pos_2d - building_center)
        in_area = dist_to_center < (self.building_area_size / 2)
        
        # Check if on table surface (not being grasped)
        on_surface = cube1_pos[2] < (self.cube_half_size + 0.02)
        not_grasped = ~self.agent.is_grasping(cube1)
        
        # Handle tensor dimensions for boolean operations
        if torch.is_tensor(not_grasped) and not_grasped.dim() > 0:
            not_grasped = not_grasped[0]
        
        return in_area & on_surface & not_grasped

    def _check_cube2_stacking(self):
        """Check if cube2 is stacked on cube1 (stability-based, no strict alignment)"""
        # Check if cube1 is placed first
        if torch.is_tensor(self.cube1_placed):
            if self.cube1_placed.dim() == 0:
                cube1_placed_val = self.cube1_placed.item()
            else:
                cube1_placed_val = self.cube1_placed[0].item()
        else:
            cube1_placed_val = bool(self.cube1_placed)
        
        if not cube1_placed_val:
            return torch.tensor(False, device=self.device)
            
        cube1 = self.objects["cube1"]
        cube2 = self.objects["cube2"]
        
        # Get positions and handle dimensions
        cube1_pos = cube1.pose.p
        cube2_pos = cube2.pose.p
        
        if torch.is_tensor(cube1_pos) and cube1_pos.dim() > 1:
            cube1_pos = cube1_pos[0]
        if torch.is_tensor(cube2_pos) and cube2_pos.dim() > 1:
            cube2_pos = cube2_pos[0]
        
        # Check if cube2 is higher than cube1 (stacked on top)
        height_check = cube2_pos[2] > cube1_pos[2] + self.cube_half_size
        
        # Check reasonable proximity (not too far away horizontally)
        # Extract 2D positions properly
        cube1_pos_2d = cube1_pos[:2]
        cube2_pos_2d = cube2_pos[:2]
        horizontal_distance = torch.linalg.norm(cube2_pos_2d - cube1_pos_2d)
        proximity_check = horizontal_distance < (2 * self.cube_half_size + 0.02)  # Lenient proximity
        
        # Check not being grasped
        not_grasped = ~self.agent.is_grasping(cube2)
        if torch.is_tensor(not_grasped) and not_grasped.dim() > 0:
            not_grasped = not_grasped[0]
        
        return height_check & proximity_check & not_grasped

    def _check_triangle_placement(self):
        """Check if triangle is placed on top of cube tower"""
        # Check if cube2 is stacked first
        if torch.is_tensor(self.cube2_stacked):
            if self.cube2_stacked.dim() == 0:
                cube2_stacked_val = self.cube2_stacked.item()
            else:
                cube2_stacked_val = self.cube2_stacked[0].item()
        else:
            cube2_stacked_val = bool(self.cube2_stacked)
        
        if not cube2_stacked_val:
            return torch.tensor(False, device=self.device)
                
        cube2 = self.objects["cube2"]
        triangle = self.objects["triangle"]
        
        # Get positions and handle dimensions
        cube2_pos = cube2.pose.p
        triangle_pos = triangle.pose.p
        
        if torch.is_tensor(cube2_pos) and cube2_pos.dim() > 1:
            cube2_pos = cube2_pos[0]
        if torch.is_tensor(triangle_pos) and triangle_pos.dim() > 1:
            triangle_pos = triangle_pos[0]
        
        # Check if triangle is higher than cube2 (on top of stack)
        height_check = triangle_pos[2] > cube2_pos[2] + self.cube_half_size/2
        
        # Check reasonable proximity (not too far away horizontally)
        # Extract 2D positions properly
        cube2_pos_2d = cube2_pos[:2]
        triangle_pos_2d = triangle_pos[:2]
        horizontal_distance = torch.linalg.norm(triangle_pos_2d - cube2_pos_2d)
        proximity_check = horizontal_distance < (2 * self.cube_half_size + 0.02)  # Lenient proximity
        
        # Check not being grasped
        not_grasped = ~self.agent.is_grasping(triangle)
        if torch.is_tensor(not_grasped) and not_grasped.dim() > 0:
            not_grasped = not_grasped[0]
        
        return height_check & proximity_check & not_grasped

    def _check_structure_stability(self):
        """Check if the entire structure is stable (not moving)"""
        stable = torch.tensor(True, device=self.device) 
        
        for obj_name, obj in self.objects.items():
            current_pos = obj.pose.p
            if torch.is_tensor(current_pos):
                if current_pos.dim() > 1:
                    current_pos = current_pos[0]
            
            prev_pos = self.prev_positions[obj_name]
            if torch.is_tensor(prev_pos):
                if prev_pos.dim() > 1:
                    prev_pos = prev_pos[0]
            
            # Check if object has moved significantly
            movement = torch.linalg.norm(current_pos - prev_pos)
            if movement > self.stability_threshold:
                stable = torch.tensor(False, device=self.device)
                break
                
            # Update previous position
            if torch.is_tensor(self.prev_positions[obj_name]):
                if self.prev_positions[obj_name].dim() > 1:
                    self.prev_positions[obj_name][0] = current_pos.clone()
                else:
                    self.prev_positions[obj_name] = current_pos.clone()
            else:
                self.prev_positions[obj_name] = current_pos.clone()
        
        return stable

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        

        cube1_reward = self._compute_cube1_placement_reward()
        
        # 2. Reward for stacking cube2 on cube1
        cube2_reward = self._compute_cube2_stacking_reward()
        
        # 3. Reward for placing triangle on top
        triangle_reward = self._compute_triangle_placement_reward()
        
        reward = cube1_reward + cube2_reward + triangle_reward
        
        # Large bonus for complete success
        success = info["success"]
        if torch.is_tensor(success):
            reward = torch.where(success, reward + 5.0, reward)
        else:
            if success:
                reward = reward + 5.0
        
        return reward

    def _compute_cube2_stacking_reward(self):
        """Reward for stacking cube2 on cube1"""
        # Safe check for cube1_placed
        if torch.is_tensor(self.cube1_placed):
            if self.cube1_placed.dim() == 0:
                cube1_placed_val = self.cube1_placed.item()
            else:
                cube1_placed_val = self.cube1_placed[0].item()
        else:
            cube1_placed_val = bool(self.cube1_placed)
        
        if not cube1_placed_val:
            return torch.tensor(0.0, device=self.device)
            
        cube1 = self.objects["cube1"]
        cube2 = self.objects["cube2"]
        
        cube1_pos = cube1.pose.p
        cube2_pos = cube2.pose.p
        
        # Handle batch dimensions
        if cube1_pos.dim() > 1:
            cube1_pos = cube1_pos[0]
        if cube2_pos.dim() > 1:
            cube2_pos = cube2_pos[0]
        
        # Reward for being above cube1
        height_diff = cube2_pos[2] - cube1_pos[2]
        height_reward = torch.clamp(height_diff / (2 * self.cube_half_size), 0, 2)
        
        # Reward for horizontal proximity (lenient)
        horizontal_dist = torch.linalg.norm(cube2_pos[:2] - cube1_pos[:2])
        proximity_reward = 1.0 - torch.tanh(2.0 * horizontal_dist)  # More lenient
        
        # Bonus for successful stacking
        stacking_bonus = self.cube2_stacked.float() * 2.0
        if stacking_bonus.dim() > 0:
            stacking_bonus = stacking_bonus[0]
        
        return height_reward + proximity_reward + stacking_bonus

    def _compute_triangle_placement_reward(self):
        """Reward for placing triangle on top of cube tower"""
        # Safe check for cube2_stacked
        if torch.is_tensor(self.cube2_stacked):
            if self.cube2_stacked.dim() == 0:
                cube2_stacked_val = self.cube2_stacked.item()
            else:
                cube2_stacked_val = self.cube2_stacked[0].item()
        else:
            cube2_stacked_val = bool(self.cube2_stacked)
        
        if not cube2_stacked_val:
            return torch.tensor(0.0, device=self.device)
            
        cube2 = self.objects["cube2"]
        triangle = self.objects["triangle"]
        
        cube2_pos = cube2.pose.p
        triangle_pos = triangle.pose.p
        
        # Handle batch dimensions
        if cube2_pos.dim() > 1:
            cube2_pos = cube2_pos[0]
        if triangle_pos.dim() > 1:
            triangle_pos = triangle_pos[0]
        
        # Reward for being above cube2
        height_diff = triangle_pos[2] - cube2_pos[2]
        height_reward = torch.clamp(height_diff / (self.cube_half_size + self.triangle_height), 0, 2)
        
        # Reward for horizontal proximity (lenient)
        horizontal_dist = torch.linalg.norm(triangle_pos[:2] - cube2_pos[:2])
        proximity_reward = 1.0 - torch.tanh(2.0 * horizontal_dist)  # More lenient
        
        # Bonus for successful placement
        placement_bonus = self.triangle_placed.float() * 2.0
        if placement_bonus.dim() > 0:
            placement_bonus = placement_bonus[0]
        
        return height_reward + proximity_reward + placement_bonus

    def _compute_cube1_placement_reward(self):
        """Reward for moving cube1 towards and placing it in building area"""
        cube1 = self.objects["cube1"]
        cube1_pos = cube1.pose.p
        
        # Handle batch dimensions
        if cube1_pos.dim() > 1:
            cube1_pos = cube1_pos[0]
        
        # Get 2D position
        cube1_pos_2d = cube1_pos[:2]
        
        # Distance to building center
        dist_to_center = torch.linalg.norm(cube1_pos_2d - self.building_center)
        proximity_reward = 1.0 - torch.tanh(3.0 * dist_to_center)
        
        # Bonus for actual placement
        placement_bonus = self.cube1_placed.float() * 2.0
        if placement_bonus.dim() > 0:
            placement_bonus = placement_bonus[0]
        
        return proximity_reward + placement_bonus

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 12.0
    
    def _compute_sequential_bonus(self, info: Dict):
        """Bonus for following correct building sequence"""
        progress = info["building_progress"]
        
        # Handle tensor dimensions properly
        if torch.is_tensor(progress):
            if progress.dim() > 0:
                progress = progress[0]
            progress_val = progress.item()
        else:
            progress_val = float(progress)
        
        # Encourage gradual progress
        if progress_val > 0.9:  # Nearly complete
            return torch.tensor(1.0, device=self.device)
        elif progress_val > 0.6:  # Two pieces placed
            return torch.tensor(0.5, device=self.device)
        elif progress_val > 0.3:  # One piece placed
            return torch.tensor(0.2, device=self.device)
        else:
            return torch.tensor(0.0, device=self.device)

SpatialHardBuildBlockEnv.__doc__ = SPATIAL_HARD_BUILD_DOC_STRING