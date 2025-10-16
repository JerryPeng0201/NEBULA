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
from typing import Any, Dict, Union, List

@register_env("Spatial-PickCube-Medium", max_episode_steps=75)
class SpatialMediumPickCubeEnv(BaseEnv):
    """Task Description:
    Pick cubes based on complex 3D spatial relationships involving multiple reference objects. 
    The robot must understand relations like "inside", "outside", "on top of", "beside" relative to containers and other objects.

    Spatial Relations:
    - Inside/Outside: Relative to container objects  
    - On top of/Beside: Vertical and horizontal relationships between objects
    - Multiple reference objects create complex spatial scenes

    Randomizations:
    - Container and object positions are randomized in safe areas
    - Multiple objects of different colors
    - Spatial relationships vary between episodes

    Success Conditions:
    - Pick the correct object satisfying the spatial description
    - Object is successfully grasped and lifted
    - Robot maintains stability throughout
    """
    
    SUPPORTED_ROBOTS = ["panda", "fetch"]
    agent: Union[Panda, Fetch]
    
    cube_half_size = 0.02
    platform_half_size = 0.1 
    lift_thresh = 0.05
    task_instruction = "Pick the target object based on spatial relation" # update later
    
    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.spatial_relations_3d = ["inside", "outside", "on_top_of", "beside"]
        self.object_colors = {
            "red": [1, 0, 0, 1],
            "green": [0, 1, 0, 1], 
            "blue": [0, 0, 1, 1],
            "yellow": [1, 1, 0, 1],
        }
        # Available containers
        self.ycb_containers = {
            "bowl": "024_bowl"
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
        pose = sapien_utils.look_at(eye=[0.8, 0.8, 1.0], target=[0, 0, 0.2])
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
        
        # Container will be created in _initialize_episode since it varies per episode
        self.container = None
        
        # Create small objects with initial high position
        self.objects = {}
        for color_name, color_rgb in self.object_colors.items():
            obj = actors.build_cube(
                self.scene,
                half_size=self.cube_half_size,
                color=color_rgb,
                name=f"{color_name}_object",
                initial_pose=sapien.Pose(p=[0, 0, 1.0])  # Start high to drop down
            )
            self.objects[color_name] = obj
        
        # Create platform for stacking as simple geometry with lower height
        self.platform = actors.build_cube(
            self.scene,
            half_size=self.platform_half_size,
            color=[0.7, 0.7, 0.7, 1],
            name="platform",
            body_type="kinematic"  # Make platform kinematic
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            
            # Clean up previous container if it exists
            if self.container is not None:
                try:
                    self.scene.remove_actor(self.container)
                except:
                    pass
            
            # Randomly select container type for this episode
            self.current_container_type = random.choice(list(self.ycb_containers.keys()))
            self.current_container_ycb_id = self.ycb_containers[self.current_container_type]
            
            # Create container using actors.get_actor_builder
            container_builder = actors.get_actor_builder(
                self.scene, 
                id=f"ycb:{self.current_container_ycb_id}"
            )
            container_builder.initial_pose = sapien.Pose(p=[0, 0, 1.0])  # Start high
            self.container = container_builder.build(name=f"container_{random.randint(1000, 9999)}")
            
            # Set physics properties for stability
            if self.container.has_collision_shapes:
                for shape in self.container._bodies[0].get_collision_shapes():
                    material = shape.get_physical_material()
                    material.static_friction = 0.8
                    material.dynamic_friction = 0.7
                    material.restitution = 0.1  # Low bounce
            
            # Select spatial relation and target object
            self.current_relation_3d = random.choice(self.spatial_relations_3d)
            self.target_object = random.choice(list(self.objects.keys()))
            self.task_instruction = self._generate_task_instruction()
            
            # Position container in safe area away from robot
            container_xyz = torch.zeros((b, 3))
            container_xyz[:, :2] = torch.tensor([0.15, 0.12]) + (torch.rand((b, 2)) * 2 - 1) * 0.02
            container_xyz[:, 2] = self._get_container_base_height()
            self.container.set_pose(Pose.create_from_pq(container_xyz))
            
            # Position platform in safe area  
            platform_xyz = torch.zeros((b, 3))
            platform_xyz[:, :2] = torch.tensor([0.20, -0.10]) + (torch.rand((b, 2)) * 2 - 1) * 0.02
            platform_xyz[:, 2] = -0.05
            self.platform.set_pose(Pose.create_from_pq(platform_xyz))
            
            # Position target object based on spatial relation to ensure clear spatial relationships
            target_xyz = torch.zeros((b, 3))
            
            if self.current_relation_3d == "inside":
                # Place target object clearly inside the container
                target_xyz[:, :2] = container_xyz[:, :2] + (torch.rand((b, 2)) * 2 - 1) * 0.015  # Smaller random offset
                target_xyz[:, 2] = container_xyz[:, 2] + self._get_container_height() * 0.3 + self.cube_half_size  # Lower in container
            elif self.current_relation_3d == "outside":
                # Place target object clearly outside (not in container, not on platform)
                target_xyz[:, :2] = torch.tensor([0.05, 0.25]) + (torch.rand((b, 2)) * 2 - 1) * 0.02  # Specific outside position
                target_xyz[:, 2] = self.cube_half_size
            elif self.current_relation_3d == "on_top_of":
                target_xyz[:, :2] = platform_xyz[:, :2] + (torch.rand((b, 2)) * 2 - 1) * 0.03
                target_xyz[:, 2] = platform_xyz[:, 2] + self.platform_half_size + self.cube_half_size  # on top of platform
            else:  # beside
                target_xyz[:, :2] = platform_xyz[:, :2] + torch.tensor([0.08, 0.0])  # beside platform
                target_xyz[:, 2] = self.cube_half_size
            
            self.objects[self.target_object].set_pose(Pose.create_from_pq(target_xyz))
            
            # Position remaining objects with clear spatial logic - ensure only target object satisfies the relation
            remaining_objects = [k for k in self.objects.keys() if k != self.target_object]
            
            if self.current_relation_3d == "outside":
                # For outside: target is the ONLY object truly "outside" (not in container)
                # Put ALL other objects inside container to avoid any ambiguity
                for i, color in enumerate(remaining_objects):
                    obj_xyz = torch.zeros((b, 3))
                    # All other objects go inside container
                    obj_xyz[:, :2] = container_xyz[:, :2] + (torch.rand((b, 2)) * 2 - 1) * 0.015
                    obj_xyz[:, 2] = container_xyz[:, 2] + self._get_container_height() * (0.3 + i * 0.1) + self.cube_half_size  # Stack at different heights
                    self.objects[color].set_pose(Pose.create_from_pq(obj_xyz))
            
            else:
                # For all other relations: place other objects in neutral positions
                neutral_positions = [
                    [0.30, 0.10],    # Far right neutral area  
                    [0.35, -0.25],   # Far back-right neutral area
                    [-0.05, -0.30]   # Back-left neutral area
                ]
                
                for i, color in enumerate(remaining_objects):
                    obj_xyz = torch.zeros((b, 3))
                    pos = neutral_positions[i % len(neutral_positions)]
                    
                    # Add extra checks to ensure neutral positions don't conflict with target relation
                    if self.current_relation_3d == "beside":
                        # Make sure neutral objects are far enough from platform to avoid "beside" ambiguity
                        platform_pos = platform_xyz[0, :2].cpu().numpy()
                        distance_to_platform = np.linalg.norm(np.array(pos) - platform_pos)
                        if distance_to_platform < 0.15:  # Too close to be clearly "not beside"
                            # Move to a more distant neutral position
                            pos = [0.40, 0.15] if i % 2 == 0 else [-0.10, 0.25]
                    
                    obj_xyz[:, :2] = torch.tensor(pos) + (torch.rand((b, 2)) * 2 - 1) * 0.02
                    obj_xyz[:, 2] = self.cube_half_size
                    self.objects[color].set_pose(Pose.create_from_pq(obj_xyz))
            
            # Let physics settle before proceeding
            for _ in range(15):  # Longer settling time for complex objects
                self.scene.step()
                # Ensure container stays stable
                self.container.set_linear_velocity(torch.zeros((b, 3)))
                self.container.set_angular_velocity(torch.zeros((b, 3)))

    def _get_container_base_height(self):
        """Get the appropriate base height for different container types"""
        if self.current_container_type == "bowl":
            return 0.04  # Bowls sit lower
        else:
            return 0.05  # Default

    def _get_container_height(self):
        """Get the internal height of different container types for inside relationships"""
        if self.current_container_type == "bowl":
            return 0.06  # Bowl depth
        else:
            return 0.06  # Default

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            spatial_relation_3d_encoded=self._encode_spatial_relation_3d(self.current_relation_3d),
            target_object_encoded=self._encode_target_object(self.target_object),
            container_type_encoded=self._encode_container_type(self.current_container_type),
            task_type_encoded=self._encode_task_type("pick_3d_spatial"),
        )
        
        if "state" in self.obs_mode:
            target_obj = self.objects[self.target_object]
            
            obs.update(
                container_pose=self.container.pose.raw_pose,
                platform_pose=self.platform.pose.raw_pose,
                target_object_pose=target_obj.pose.raw_pose,
                tcp_to_target_pos=target_obj.pose.p - self.agent.tcp.pose.p,
                tcp_to_container_pos=self.container.pose.p - self.agent.tcp.pose.p,
                tcp_to_platform_pos=self.platform.pose.p - self.agent.tcp.pose.p,
                target_to_container_pos=self.container.pose.p - target_obj.pose.p,
                target_to_platform_pos=self.platform.pose.p - target_obj.pose.p,
                container_to_platform_pos=self.platform.pose.p - self.container.pose.p,
                is_grasping_target=self.agent.is_grasping(target_obj),
                container_inner_radius=torch.tensor(self._get_container_inner_radius()),
                container_height=torch.tensor(self._get_container_height()),
                platform_half_size=torch.tensor(self.platform_half_size),
            )
            
            # Add all object poses and grasping states
            for color_name, obj in self.objects.items():
                obs[f"{color_name}_object_pose"] = obj.pose.raw_pose
                obs[f"is_grasping_{color_name}"] = self.agent.is_grasping(obj)
                obs[f"{color_name}_to_container_pos"] = self.container.pose.p - obj.pose.p
                obs[f"{color_name}_to_platform_pos"] = self.platform.pose.p - obj.pose.p
                
        return obs

    def _encode_spatial_relation_3d(self, relation):
        """Encode 3D spatial relation as integer: inside=0, outside=1, on_top_of=2, beside=3"""
        relation_map = {"inside": 0, "outside": 1, "on_top_of": 2, "beside": 3}
        return relation_map.get(relation, 0)

    def _encode_target_object(self, target_object):
        """Encode target object color as integer: red=0, green=1, blue=2, yellow=3"""
        target_map = {"red": 0, "green": 1, "blue": 2, "yellow": 3}
        return target_map.get(target_object, 0)

    def _encode_container_type(self, container_type):
        """Encode container type as integer: bowl=0, pitcher=1"""
        container_map = {"bowl": 0, "pitcher": 1}
        return container_map.get(container_type, 0)

    def _encode_task_type(self, task_type):
        """Encode task type as integer: pick_3d_spatial=0, pick_closest=1, place_between=2"""
        task_map = {"pick_3d_spatial": 0, "pick_closest": 1, "place_between": 2}
        return task_map.get(task_type, 0)

    def evaluate(self):
        target_obj = self.objects[self.target_object]
        
        is_lifted = target_obj.pose.p[:, 2] > self.cube_half_size + self.lift_thresh
        is_grasped = self.agent.is_grasping(target_obj)
        is_robot_static = self.agent.is_static(0.2)
        
        success = is_lifted & is_grasped & is_robot_static
        
        return {
            "success": success,
            "is_lifted": is_lifted, 
            "is_grasped": is_grasped,
            "is_robot_static": is_robot_static,
            "spatial_relation": self.current_relation_3d,
            "target_object": self.target_object,
            "container_type": self.current_container_type,
            "container_ycb_id": self.current_container_ycb_id,
            "task_instruction": self.task_instruction,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        target_obj = self.objects[self.target_object]
        
        # 1. Reaching reward
        tcp_to_target_dist = torch.linalg.norm(
            target_obj.pose.p - self.agent.tcp_pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_target_dist)
        
        # 2. Grasping reward
        grasping_reward = self.agent.is_grasping(target_obj).float() * 1.5
        
        # 3. Lifting reward
        lift_height = target_obj.pose.p[:, 2] - self.cube_half_size
        lifting_reward = torch.clamp(lift_height / self.lift_thresh, 0, 1) * 1.5
        
        # 4. Spatial understanding bonus (adapted for different container types)
        spatial_bonus = self._compute_spatial_bonus()
        
        # 5. Wrong object penalty
        wrong_grasp_penalty = self._compute_wrong_grasp_penalty()
        
        reward = reaching_reward + grasping_reward + lifting_reward + spatial_bonus - wrong_grasp_penalty
        
        # Success bonus
        reward[info["success"]] += 3
            
        return reward

    def _compute_spatial_bonus(self):
        """Bonus for exploring the correct spatial area - adapted for different container types"""
        target_obj = self.objects[self.target_object]
        
        if self.current_relation_3d == "inside":
            # Bonus for being near container when target is inside
            container_to_tcp_dist = torch.linalg.norm(
                self.container.pose.p[:, :2] - self.agent.tcp_pose.p[:, :2], axis=1
            )
            # Scale bonus based on container size (removed pitcher scaling)
            proximity_scale = 3.0
            return (1 - torch.tanh(proximity_scale * container_to_tcp_dist)) * 0.3
        elif self.current_relation_3d == "on_top_of":
            # Bonus for being near platform when target is on top
            platform_to_tcp_dist = torch.linalg.norm(
                self.platform.pose.p[:, :2] - self.agent.tcp_pose.p[:, :2], axis=1
            )
            return (1 - torch.tanh(3 * platform_to_tcp_dist)) * 0.3
        
        return torch.zeros_like(self.agent.tcp_pose.p[:, 0])
    
    def _generate_task_instruction(self):
        """Generate a language instruction based on the current spatial relation and container type"""
        container_name = self.current_container_type
        
        if self.current_relation_3d == "inside":
            return f"Pick the object inside the {container_name}"
        elif self.current_relation_3d == "outside":
            return f"Pick the object outside the {container_name}"
        elif self.current_relation_3d == "on_top_of":
            return f"Pick the object on top of the platform"
        elif self.current_relation_3d == "beside":
            return f"Pick the object beside the platform"
        else:
            return f"Pick the target object based on spatial relation relative to the {container_name}"
        

    def _compute_wrong_grasp_penalty(self):
        """Penalty for grasping wrong objects"""
        penalty = torch.zeros_like(self.agent.tcp_pose.p[:, 0])
        for color, obj in self.objects.items():
            if color != self.target_object:
                wrong_grasp = self.agent.is_grasping(obj).float()
                penalty += wrong_grasp * 0.3
        return penalty

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 7
    
    def get_task_instruction(self):
        return self.task_instruction