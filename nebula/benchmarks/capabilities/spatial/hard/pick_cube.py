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

@register_env("Spatial-PickCube-Hard", max_episode_steps=150)
class SpatialHardPickCubeEnv(BaseEnv):
    """Task Description:
    Perform complex multi-step spatial reasoning with nested spatial relationships. 
    The robot must understand complex relations such as:
    - "Pick the green cube that is on top of the object inside the red container"
    - "Pick the blue cube that is beside the object under the platform"
    - "Pick the object that is behind the stacked items"

    This requires:
    1. Multi-step spatial reasoning (container -> inside object -> on top object)
    2. Understanding complex spatial hierarchies and relationships
    3. Identifying target objects through nested spatial descriptions

    Spatial Relations:
    - Nested relationships: "on_top_of_inside", "beside_under_platform", "behind_stacked_items"
    - Multiple reference objects creating complex spatial scenes
    - Clear spatial hierarchies with unambiguous arrangements

    Randomizations:
    - Complex nested object arrangements with 2-3 levels of spatial hierarchy
    - Multiple containers and platforms in different configurations  
    - Random spatial relationship selection for each episode

    Success Conditions:
    - Successfully identify and pick the correct target object through multi-step reasoning
    - Object is grasped and lifted above minimum threshold
    - Robot maintains stability throughout the task
    """
    
    SUPPORTED_ROBOTS = ["panda", "fetch"]
    agent: Union[Panda, Fetch]
    
    cube_half_size = 0.02
    platform_half_size = [0.08, 0.08, 0.01]  # Flat rectangular platform
    lift_thresh = 0.06

    task_instruction = ""
    
    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        
        # Expanded complex spatial relations for comprehensive spatial reasoning testing
        self.complex_relations = [
            "on_top_of_inside",      # "object on top of the item inside the container"
            "beside_under_platform", # "object beside the item under the platform"  
            "behind_stacked_items",  # "object behind the stacked items"
            "inside_beside_platform", # "object inside the container beside the platform"
            "under_behind_container", # "object under the platform behind the container"
            "on_top_of_beside_container", # "object on top of the item beside the container"
        ]
        
        # Colors for different object types
        self.object_colors = {
            "red": [1, 0, 0, 1],
            "green": [0, 1, 0, 1], 
            "blue": [0, 0, 1, 1],
            "yellow": [1, 1, 0, 1],
        }
        
        # Available containers (using YCB models)
        self.ycb_containers = {
            "plate": "029_plate",
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
        pose = sapien_utils.look_at(eye=[1.2, 1.2, 1.5], target=[0, 0, 0.25])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.table_scene.build()
        
        # Ensure assets are available
        from nebula import ASSET_DIR
        if not os.path.exists(ASSET_DIR / "assets/mani_skill2_ycb"):
            from nebula.utils import assets, download_asset
            download_asset.download(assets.DATA_SOURCES["ycb"])
        
        # Container will be created dynamically in _initialize_episode
        self.container = None
        
        # Create 4 manipulable objects with enhanced stability
        self.objects = {}
        for color_name, color_rgb in self.object_colors.items():
            obj = actors.build_cube(
                self.scene,
                half_size=self.cube_half_size,
                color=color_rgb,
                name=f"{color_name}_object",
                initial_pose=sapien.Pose(p=[0, 0, 1.0])  # Start high above table to drop down
            )
            
            # Set physics properties for stability
            if obj.has_collision_shapes:
                for shape in obj._bodies[0].get_collision_shapes():
                    material = shape.get_physical_material()
                    material.static_friction = 0.9
                    material.dynamic_friction = 0.8
                    material.restitution = 0.05  # Very low bounce
            
            self.objects[color_name] = obj
        
        # Create platform as flat rectangle floating above table using SAPIEN builder
        platform_builder = self.scene.create_actor_builder()
        platform_builder.add_box_collision(half_size=[0.08, 0.08, 0.01])  # Flat rectangle
        platform_builder.add_box_visual(half_size=[0.10, 0.10, 0.01], 
                                       material=sapien.render.RenderMaterial(base_color=[0.7, 0.7, 0.7, 1]))
        platform_builder.set_physx_body_type("kinematic")
        platform_builder.initial_pose = sapien.Pose(p=[0, 0, 2.5])  # Start high
        self.platform = platform_builder.build(name="platform")

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            
            actors_to_remove = []
            
            if hasattr(self, 'container') and self.container is not None:
                actors_to_remove.append(self.container)
                
            for actor in actors_to_remove:
                try:
                    self.scene.remove_actor(actor)
                except Exception as e:
                    print(f"Warning: Failed to remove actor {actor.name}: {e}")
                    
            self.current_container_type = random.choice(list(self.ycb_containers.keys()))
            container_builder = actors.get_actor_builder(
                self.scene, 
                id=f"ycb:{self.ycb_containers[self.current_container_type]}"
            )
            container_builder.initial_pose = sapien.Pose(p=[0, 0, 1.0])
            self.container = container_builder.build(name=f"container_{self.current_container_type}")
            
            # Set physics properties for stability
            if self.container.has_collision_shapes:
                for shape in self.container._bodies[0].get_collision_shapes():
                    material = shape.get_physical_material()
                    material.static_friction = 0.8
                    material.dynamic_friction = 0.7
                    material.restitution = 0.1  # Low bounce
            
            # Sample complex spatial relationship for this episode
            self.current_complex_relation = random.choice(self.complex_relations)
            # Choose objects for the spatial arrangement
            available_colors = list(self.objects.keys())
            self.target_object_color = random.choice(available_colors)
            self.intermediate_object_color = random.choice([k for k in available_colors 
                                                          if k != self.target_object_color])
            
            # Position container and platform
            self._position_container_and_platform(b)
            
            # Set up complex spatial arrangements based on the selected relation
            self._setup_spatial_arrangement(b)
            
            # Position remaining objects as distractors in neutral positions
            self._position_distractor_objects(b)

            for i in range(5):
                self.scene.step()

    def _position_container_and_platform(self, b):
        """Position container and platform with separation"""
        # Position container
        container_xyz = torch.zeros((b, 3))
        container_xyz[:, :2] = torch.tensor([0.15, 0.15]) + (torch.rand((b, 2)) * 2 - 1) * 0.02
        container_xyz[:, 2] = self._get_container_base_height()
        self.container.set_pose(Pose.create_from_pq(container_xyz))
        
        # Position platform floating above table surface
        platform_xyz = torch.zeros((b, 3))
        platform_xyz[:, :2] = torch.tensor([0.20, -0.10]) + (torch.rand((b, 2)) * 2 - 1) * 0.02
        platform_xyz[:, 2] = 0.08  # Floating height above table
        self.platform.set_pose(Pose.create_from_pq(platform_xyz))

    def _setup_spatial_arrangement(self, b):
        """Set up complex nested spatial relationships with clear logic - ensure no ambiguity"""
        container_pos = self.container.pose.p
        platform_pos = self.platform.pose.p
        target_obj = self.objects[self.target_object_color]
        intermediate_obj = self.objects[self.intermediate_object_color]
        
        # Initialize stack_objects tracking
        self.stack_objects = set()
        
        if self.current_complex_relation == "on_top_of_inside":
            # Intermediate object inside container
            intermediate_xyz = torch.zeros((b, 3))
            intermediate_xyz[:, :2] = container_pos[:, :2] + (torch.rand((b, 2)) * 2 - 1) * 0.015
            intermediate_xyz[:, 2] = container_pos[:, 2] + self._get_container_height() * 0.3 + self.cube_half_size
            intermediate_obj.set_pose(Pose.create_from_pq(intermediate_xyz))
            
            # Target object on top of intermediate object with stable positioning
            target_xyz = intermediate_xyz.clone()
            target_xyz[:, 2] += self.cube_half_size * 2.1  # Slightly higher for stability
            target_obj.set_pose(Pose.create_from_pq(target_xyz))
            
            # Immediately stabilize the stacked objects
            intermediate_obj.set_linear_velocity(torch.zeros((b, 3)))
            intermediate_obj.set_angular_velocity(torch.zeros((b, 3)))
            target_obj.set_linear_velocity(torch.zeros((b, 3)))
            target_obj.set_angular_velocity(torch.zeros((b, 3)))
            
            # Mark objects used in this arrangement
            self.used_objects = {self.target_object_color, self.intermediate_object_color}
            
        elif self.current_complex_relation == "beside_under_platform":
            # Intermediate object under the floating platform
            intermediate_xyz = torch.zeros((b, 3))
            intermediate_xyz[:, :2] = platform_pos[:, :2] + (torch.rand((b, 2)) * 2 - 1) * 0.02
            intermediate_xyz[:, 2] = self.cube_half_size  # On table surface, under the floating platform
            intermediate_obj.set_pose(Pose.create_from_pq(intermediate_xyz))
            
            # Target object beside the object under the platform (NOT beside the platform itself)
            target_xyz = torch.zeros((b, 3))
            target_xyz[:, :2] = intermediate_xyz[:, :2] + torch.tensor([0.06, 0.01])  # Close beside the intermediate object
            target_xyz[:, 2] = self.cube_half_size  # On table surface, same level as intermediate
            target_obj.set_pose(Pose.create_from_pq(target_xyz))
            
            # Mark objects used in this arrangement
            self.used_objects = {self.target_object_color, self.intermediate_object_color}
            
        elif self.current_complex_relation == "behind_stacked_items":
            # Create a proper stack with intermediate object at bottom
            stack_xyz = torch.zeros((b, 3))
            stack_xyz[:, :2] = torch.tensor([0.10, 0.05]) + (torch.rand((b, 2)) * 2 - 1) * 0.02
            stack_xyz[:, 2] = self.cube_half_size
            intermediate_obj.set_pose(Pose.create_from_pq(stack_xyz))
            
            # Find another object to create the stack (not target, not intermediate)
            remaining_objects = [k for k in self.objects.keys() 
                               if k not in {self.target_object_color, self.intermediate_object_color}]
            
            if remaining_objects:
                stack_top_color = remaining_objects[0]
                stack_top_obj = self.objects[stack_top_color]
                
                # Place the second object on top to create a real stack with stability
                stack_top_xyz = stack_xyz.clone()
                stack_top_xyz[:, 2] += self.cube_half_size * 2.1  # Slightly higher for stability
                stack_top_obj.set_pose(Pose.create_from_pq(stack_top_xyz))
                
                # Immediately stabilize the stack
                intermediate_obj.set_linear_velocity(torch.zeros((b, 3)))
                intermediate_obj.set_angular_velocity(torch.zeros((b, 3)))
                stack_top_obj.set_linear_velocity(torch.zeros((b, 3)))
                stack_top_obj.set_angular_velocity(torch.zeros((b, 3)))
                
                # Mark stack objects
                self.stack_objects = {self.intermediate_object_color, stack_top_color}
                
                # Target object behind the stack (from robot's perspective - robot is at [-0.615, 0, 0])
                # Behind means further away from robot (more positive X direction)
                target_xyz = stack_xyz.clone()
                target_xyz[:, 0] += 0.08  # Behind = further from robot (positive X direction)
                target_xyz[:, 1] += 0.02  # Slightly offset in Y
                target_obj.set_pose(Pose.create_from_pq(target_xyz))
                
                # Mark all objects used in this arrangement
                self.used_objects = {self.target_object_color, self.intermediate_object_color, stack_top_color}
            else:
                # Fallback: if not enough objects, just place target behind intermediate
                self.stack_objects = {self.intermediate_object_color}
                target_xyz = stack_xyz.clone()
                target_xyz[:, 0] += 0.08  # Behind = further from robot
                target_xyz[:, 1] += 0.02  # Slightly offset
                target_obj.set_pose(Pose.create_from_pq(target_xyz))
                
                # Mark objects used
                self.used_objects = {self.target_object_color, self.intermediate_object_color}
                
        elif self.current_complex_relation == "inside_beside_platform":
            # First, position container beside platform
            container_beside_xyz = torch.zeros((b, 3))
            container_beside_xyz[:, :2] = platform_pos[:, :2] + torch.tensor([0.15, 0.02])  # Beside platform
            container_beside_xyz[:, 2] = self._get_container_base_height()
            self.container.set_pose(Pose.create_from_pq(container_beside_xyz))
            
            # Target object inside the container (which is beside platform)
            target_xyz = torch.zeros((b, 3))
            target_xyz[:, :2] = container_beside_xyz[:, :2] + (torch.rand((b, 2)) * 2 - 1) * 0.015
            target_xyz[:, 2] = container_beside_xyz[:, 2] + self._get_container_height() * 0.3 + self.cube_half_size
            target_obj.set_pose(Pose.create_from_pq(target_xyz))
            
            # Immediately stabilize target object in container
            target_obj.set_linear_velocity(torch.zeros((b, 3)))
            target_obj.set_angular_velocity(torch.zeros((b, 3)))
            
            # Intermediate object as distractor in neutral position (NOT used in spatial arrangement)
            intermediate_xyz = torch.zeros((b, 3))
            intermediate_xyz[:, :2] = torch.tensor([0.30, 0.15]) + (torch.rand((b, 2)) * 2 - 1) * 0.02
            intermediate_xyz[:, 2] = self.cube_half_size
            intermediate_obj.set_pose(Pose.create_from_pq(intermediate_xyz))
            
            # Stabilize intermediate object
            intermediate_obj.set_linear_velocity(torch.zeros((b, 3)))
            intermediate_obj.set_angular_velocity(torch.zeros((b, 3)))
            
            self.used_objects = {self.target_object_color, self.intermediate_object_color}
            
        elif self.current_complex_relation == "under_behind_container":
            # Target object under platform, and platform is behind container (from robot perspective)
            # First position container
            container_front_xyz = torch.zeros((b, 3))
            container_front_xyz[:, :2] = torch.tensor([0.08, 0.05])  # Closer to robot
            container_front_xyz[:, 2] = self._get_container_base_height()
            self.container.set_pose(Pose.create_from_pq(container_front_xyz))
            
            # Position platform behind container
            platform_behind_xyz = torch.zeros((b, 3))
            platform_behind_xyz[:, :2] = container_front_xyz[:, :2] + torch.tensor([0.12, 0.02])  # Behind container
            platform_behind_xyz[:, 2] = 0.08  # Floating height
            self.platform.set_pose(Pose.create_from_pq(platform_behind_xyz))
            
            # Target object under the platform
            target_xyz = torch.zeros((b, 3))
            target_xyz[:, :2] = platform_behind_xyz[:, :2] + (torch.rand((b, 2)) * 2 - 1) * 0.02
            target_xyz[:, 2] = self.cube_half_size  # On table, under platform
            target_obj.set_pose(Pose.create_from_pq(target_xyz))
            
            # Intermediate object as distractor
            intermediate_xyz = torch.zeros((b, 3))
            intermediate_xyz[:, :2] = torch.tensor([0.35, -0.20]) + (torch.rand((b, 2)) * 2 - 1) * 0.02
            intermediate_xyz[:, 2] = self.cube_half_size
            intermediate_obj.set_pose(Pose.create_from_pq(intermediate_xyz))
            
            self.used_objects = {self.target_object_color, self.intermediate_object_color}
            
        elif self.current_complex_relation == "on_top_of_beside_container":
            # Intermediate object beside container
            intermediate_xyz = torch.zeros((b, 3))
            intermediate_xyz[:, :2] = container_pos[:, :2] + torch.tensor([0.08, 0.02])  # Beside container
            intermediate_xyz[:, 2] = self.cube_half_size
            intermediate_obj.set_pose(Pose.create_from_pq(intermediate_xyz))
            
            # Target object on top of the intermediate object with stability
            target_xyz = intermediate_xyz.clone()
            target_xyz[:, 2] += self.cube_half_size * 2.1  # Slightly higher for stability
            target_obj.set_pose(Pose.create_from_pq(target_xyz))
            
            # Immediately stabilize both objects
            intermediate_obj.set_linear_velocity(torch.zeros((b, 3)))
            intermediate_obj.set_angular_velocity(torch.zeros((b, 3)))
            target_obj.set_linear_velocity(torch.zeros((b, 3)))
            target_obj.set_angular_velocity(torch.zeros((b, 3)))
            
            self.used_objects = {self.target_object_color, self.intermediate_object_color}

    def _position_distractor_objects(self, b):
        """Position remaining objects as distractors in neutral positions to avoid any spatial ambiguity"""
        # Get objects that haven't been used in the spatial arrangement
        remaining_objects = [k for k in self.objects.keys() if k not in self.used_objects]
        
        # Define far neutral positions that don't satisfy any spatial relationships
        # Adjust based on current spatial relation to avoid conflicts
        if self.current_complex_relation == "behind_stacked_items":
            far_neutral_positions = [
                [-0.05, 0.25],   # Front-left (definitely in front of stack)
                [0.05, -0.30],   # Side position (not behind stack)
            ]
        elif self.current_complex_relation in ["inside_beside_platform", "under_behind_container"]:
            # For relations involving container and platform positions, use side positions
            far_neutral_positions = [
                [0.35, 0.25],    # Far front-right
                [-0.10, -0.30],  # Far back-left
            ]
        else:
            far_neutral_positions = [
                [0.35, 0.20],    # Far front-right - not beside anything, not behind anything
                [0.40, -0.30],   # Far back-right - not beside anything, not behind anything
            ]
        
        for i, color in enumerate(remaining_objects):
            obj_xyz = torch.zeros((b, 3))
            pos = far_neutral_positions[i % len(far_neutral_positions)]
            
            # Additional checks to ensure these objects don't accidentally satisfy spatial conditions
            if self.current_complex_relation == "beside_under_platform":
                # Make sure distractor objects are FAR from the intermediate object (under platform) to avoid "beside" ambiguity
                intermediate_pos = None
                for color, obj in self.objects.items():
                    if color == self.intermediate_object_color:
                        intermediate_pos = obj.pose.p[0, :2].cpu().numpy()
                        break
                
                if intermediate_pos is not None:
                    distance_to_intermediate = np.linalg.norm(np.array(pos) - intermediate_pos)
                    if distance_to_intermediate < 0.15:  # Too close to intermediate object
                        pos = [0.45, -0.35]  # Move to very far position
                    
            elif self.current_complex_relation == "behind_stacked_items":
                # Make sure distractor objects are NOT behind the stack
                stack_pos = torch.tensor([0.10, 0.05])  # Approximate stack position
                # Behind from robot perspective means X > stack_X
                if pos[0] >= stack_pos[0]:  # Would be behind or at stack level
                    # Force to front or side positions (definitely not behind)
                    if i % 2 == 0:
                        pos = [-0.05, 0.25]  # Front-left (in front of stack)
                    else:
                        pos = [0.05, -0.30]  # Side position (not behind)
                        
            elif self.current_complex_relation == "on_top_of_inside":
                # Make sure distractor objects are not inside container or on top of anything
                container_pos = self.container.pose.p[0, :2].cpu().numpy()
                distance_to_container = np.linalg.norm(np.array(pos) - container_pos)
                if distance_to_container < 0.15:  # Too close to container
                    pos = [0.40, 0.25]  # Move far away
            
            obj_xyz[:, :2] = torch.tensor(pos) + (torch.rand((b, 2)) * 2 - 1) * 0.01  # Smaller random offset for stability
            obj_xyz[:, 2] = self.cube_half_size  # Always on table surface
            self.objects[color].set_pose(Pose.create_from_pq(obj_xyz))
            
            # Immediately stabilize the distractor object
            self.objects[color].set_linear_velocity(torch.zeros((b, 3)))
            self.objects[color].set_angular_velocity(torch.zeros((b, 3)))

    def _get_container_base_height(self):
        """Get the appropriate base height for different container types"""
        if self.current_container_type == "plate":
            return 0.013
        else:
            return 0.05  # Default

    def _get_container_height(self):
        """Get the internal height of different container types for inside relationships"""
        if self.current_container_type == "plate":
            return 0.0  # plate depth
        else:
            return 0.06  # Default

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            complex_spatial_relation_encoded=self._encode_complex_relation(self.current_complex_relation),
            target_object_color_encoded=self._encode_color(self.target_object_color),
            intermediate_object_color_encoded=self._encode_color(self.intermediate_object_color),
            container_type_encoded=self._encode_container_type(self.current_container_type),
            task_type_encoded=self._encode_task_type("spatial_hard"),
        )
        
        if "state" in self.obs_mode:
            target_obj = self.objects[self.target_object_color]
            intermediate_obj = self.objects[self.intermediate_object_color]
            
            # Add key object poses and relationships
            obs.update(
                target_object_pose=target_obj.pose.raw_pose,
                intermediate_object_pose=intermediate_obj.pose.raw_pose,
                container_pose=self.container.pose.raw_pose,
                platform_pose=self.platform.pose.raw_pose,
                tcp_to_target_pos=target_obj.pose.p - self.agent.tcp.pose.p,
                tcp_to_intermediate_pos=intermediate_obj.pose.p - self.agent.tcp.pose.p,
                tcp_to_container_pos=self.container.pose.p - self.agent.tcp.pose.p,
                tcp_to_platform_pos=self.platform.pose.p - self.agent.tcp.pose.p,
                target_to_intermediate_pos=intermediate_obj.pose.p - target_obj.pose.p,
                target_to_container_pos=self.container.pose.p - target_obj.pose.p,
                target_to_platform_pos=self.platform.pose.p - target_obj.pose.p,
                intermediate_to_container_pos=self.container.pose.p - intermediate_obj.pose.p,
                intermediate_to_platform_pos=self.platform.pose.p - intermediate_obj.pose.p,
                container_to_platform_pos=self.platform.pose.p - self.container.pose.p,
                is_grasping_target=self.agent.is_grasping(target_obj),
                is_grasping_intermediate=self.agent.is_grasping(intermediate_obj),
                container_height=torch.tensor(self._get_container_height()),
                platform_half_size=torch.tensor([0.08, 0.08, 0.01]),  # Platform dimensions
            )
            
            # Add all object poses and grasping states
            for color_name, obj in self.objects.items():
                obs[f"{color_name}_object_pose"] = obj.pose.raw_pose
                obs[f"is_grasping_{color_name}"] = self.agent.is_grasping(obj)
                obs[f"{color_name}_to_container_pos"] = self.container.pose.p - obj.pose.p
                obs[f"{color_name}_to_platform_pos"] = self.platform.pose.p - obj.pose.p
            
            # Add geometric constraints
            obs.update(
                cube_half_size=torch.tensor(self.cube_half_size, device=self.device),
                lift_thresh=torch.tensor(self.lift_thresh, device=self.device),
            )
                
        return obs

    def _encode_complex_relation(self, relation):
        """Encode complex spatial relation as integer"""
        relation_map = {
            "on_top_of_inside": 0,
            "beside_under_platform": 1,
            "behind_stacked_items": 2,
            "inside_beside_platform": 3,
            "under_behind_container": 4,
            "on_top_of_beside_container": 5,
        }
        return relation_map.get(relation, 0)

    def _encode_color(self, color):
        """Encode color as integer: red=0, green=1, blue=2, yellow=3"""
        color_map = {"red": 0, "green": 1, "blue": 2, "yellow": 3}
        return color_map.get(color, 0)

    def _encode_container_type(self, container_type):
        """Encode container type as integer: plate=0"""
        container_map = {"plate": 0}
        return container_map.get(container_type, 0)

    def _encode_task_type(self, task_type):
        """Encode task type as integer: spatial_hard=0, kitchen_assembly=1, spatial_medium=2"""
        task_map = {"spatial_hard": 0, "kitchen_assembly": 1, "spatial_medium": 2}
        return task_map.get(task_type, 0)

    def _generate_task_instruction(self):
        """Generate natural language instruction for complex spatial reasoning - following task description examples"""
        container_name = self.current_container_type
        
        if self.current_complex_relation == "on_top_of_inside":
            # "Pick the cube that is on top of the object inside the red container"
            return f"Pick the cube that is on top of the object inside the {container_name}"
        elif self.current_complex_relation == "beside_under_platform":
            # "Pick the cube that is beside the object under the platform" 
            return f"Pick the cube that is beside the object under the platform"
        elif self.current_complex_relation == "behind_stacked_items":
            # "Pick the object that is behind the stacked items"
            return f"Pick the object that is behind the stacked items"
        elif self.current_complex_relation == "inside_beside_platform":
            # "Pick the object inside the container beside the platform"
            return f"Pick the object inside the {container_name} beside the platform"
        elif self.current_complex_relation == "under_behind_container":
            # "Pick the object under the platform behind the container"
            return f"Pick the object under the platform behind the {container_name}"
        elif self.current_complex_relation == "on_top_of_beside_container":
            # "Pick the object on top of the item beside the container"
            return f"Pick the object on top of the item beside the {container_name}"
        else:
            # This should never happen if complex_relations list is properly maintained
            raise ValueError(f"Unknown spatial relation: {self.current_complex_relation}. "
                        f"Available relations: {self.complex_relations}")

    def evaluate(self):
        target_obj = self.objects[self.target_object_color]
        
        is_lifted = target_obj.pose.p[:, 2] > self.cube_half_size + self.lift_thresh
        is_grasped = self.agent.is_grasping(target_obj)
        is_robot_static = self.agent.is_static(0.2)
        
        success = is_lifted & is_grasped & is_robot_static
        
        return {
            "success": success,
            "is_lifted": is_lifted,
            "is_grasped": is_grasped,
            "is_robot_static": is_robot_static,
            "spatial_relation": self.current_complex_relation,
            "target_color": self.target_object_color,
            "container_type": self.current_container_type,
            "instruction": self._generate_task_instruction()
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        target_obj = self.objects[self.target_object_color]
        
        # 1. Reaching reward (approach target object)
        tcp_to_target_dist = torch.linalg.norm(
            target_obj.pose.p - self.agent.tcp_pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_target_dist)
        
        # 2. Grasping reward
        grasping_reward = self.agent.is_grasping(target_obj).float() * 2
        
        # 3. Lifting reward
        lift_height = target_obj.pose.p[:, 2] - self.cube_half_size
        lifting_reward = torch.clamp(lift_height / self.lift_thresh, 0, 1) * 2
        
        # 4. Spatial reasoning bonus
        spatial_bonus = self._compute_spatial_reasoning_bonus()
        
        # 5. Penalty for grasping wrong objects
        wrong_grasp_penalty = self._compute_wrong_grasp_penalty()
        
        reward = reaching_reward + grasping_reward + lifting_reward + spatial_bonus - wrong_grasp_penalty
        
        # 6. Large success bonus
        reward[info["success"]] += 5
        
        return reward

    def _compute_spatial_reasoning_bonus(self):
        """Bonus for demonstrating understanding of complex spatial relationships"""
        target_obj = self.objects[self.target_object_color]
        
        # Check if robot is exploring the correct spatial area
        if self.current_complex_relation == "on_top_of_inside":
            # Bonus if tcp is near the container area where target should be
            container_to_tcp_dist = torch.linalg.norm(
                self.container.pose.p[:, :2] - self.agent.tcp_pose.p[:, :2], axis=1
            )
            return (1 - torch.tanh(3 * container_to_tcp_dist)) * 0.5
        elif self.current_complex_relation == "beside_under_platform":
            # Bonus if tcp is near the platform area
            platform_to_tcp_dist = torch.linalg.norm(
                self.platform.pose.p[:, :2] - self.agent.tcp_pose.p[:, :2], axis=1
            )
            return (1 - torch.tanh(3 * platform_to_tcp_dist)) * 0.5
        
        return torch.zeros_like(self.agent.tcp_pose.p[:, 0])

    def _compute_wrong_grasp_penalty(self):
        """Penalty for grasping incorrect objects"""
        penalty = torch.zeros_like(self.agent.tcp_pose.p[:, 0])
        for color, obj in self.objects.items():
            if color != self.target_object_color:
                wrong_grasp = self.agent.is_grasping(obj).float()
                penalty += wrong_grasp * 0.5
        return penalty

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 10
    
    def get_task_instruction(self):
        return self._generate_task_instruction()