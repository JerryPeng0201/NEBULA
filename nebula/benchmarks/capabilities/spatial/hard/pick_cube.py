from typing import Any, Dict, Union
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
    """Task: Pick target object based on complex spatial relationships.
    
    Spatial Relations:
    - on_top_of_inside: object on top of item inside container
    - behind_stacked_items: object behind stacked items
    - inside_beside_platform: object inside container beside platform
    - on_top_of_beside_container: object on top of item beside container
    """
    
    SUPPORTED_ROBOTS = ["panda", "fetch"]
    agent: Union[Panda, Fetch]
    
    cube_half_size = 0.02
    platform_half_size = [0.08, 0.08, 0.01]
    lift_thresh = 0.06
    task_instruction = ""
    
    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        
        self.complex_relations = [
            "on_top_of_inside",
            "in_front_of_stacked_items",
            "inside_beside_platform",
            "on_top_of_beside_container",
        ]
        
        self.object_colors = {
            "red": [1, 0, 0, 1],
            "green": [0, 1, 0, 1], 
            "blue": [0, 0, 1, 1],
            "yellow": [1, 1, 0, 1],
        }
        
        self.ycb_containers = {"plate": "029_plate"}
        
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
        
        from nebula import ASSET_DIR
        if not os.path.exists(ASSET_DIR / "assets/mani_skill2_ycb"):
            from nebula.utils import assets, download_asset
            download_asset.download(assets.DATA_SOURCES["ycb"])
        
        self.container = None
        
        self.objects = {}
        for color_name, color_rgb in self.object_colors.items():
            obj = actors.build_cube(
                self.scene, half_size=self.cube_half_size, color=color_rgb,
                name=f"{color_name}_object", initial_pose=sapien.Pose(p=[0, 0, 1.0])
            )
            
            if obj.has_collision_shapes:
                for shape in obj._bodies[0].get_collision_shapes():
                    material = shape.get_physical_material()
                    material.static_friction = 0.9
                    material.dynamic_friction = 0.8
                    material.restitution = 0.05
            
            self.objects[color_name] = obj
        
        platform_builder = self.scene.create_actor_builder()
        platform_builder.add_box_collision(half_size=[0.08, 0.08, 0.01])
        platform_builder.add_box_visual(half_size=[0.10, 0.10, 0.01], 
                                       material=sapien.render.RenderMaterial(base_color=[0.7, 0.7, 0.7, 1]))
        platform_builder.set_physx_body_type("kinematic")
        platform_builder.initial_pose = sapien.Pose(p=[0, 0, 2.5])
        self.platform = platform_builder.build(name="platform")

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            
            if hasattr(self, 'container') and self.container is not None:
                try:
                    self.scene.remove_actor(self.container)
                except Exception:
                    pass
                    
            self.current_container_type = random.choice(list(self.ycb_containers.keys()))
            container_builder = actors.get_actor_builder(
                self.scene, id=f"ycb:{self.ycb_containers[self.current_container_type]}"
            )
            container_builder.initial_pose = sapien.Pose(p=[0, 0, 1.0])
            self.container = container_builder.build(name=f"container_{self.current_container_type}")
            
            if self.container.has_collision_shapes:
                for shape in self.container._bodies[0].get_collision_shapes():
                    material = shape.get_physical_material()
                    material.static_friction = 0.8
                    material.dynamic_friction = 0.7
                    material.restitution = 0.1
            
            self.current_complex_relation = random.choice(self.complex_relations)
            available_colors = list(self.objects.keys())
            self.target_object_color = random.choice(available_colors)
            self.intermediate_object_color = random.choice([k for k in available_colors 
                                                          if k != self.target_object_color])
            
            self._position_container_and_platform(b)
            self._setup_spatial_arrangement(b)
            self._position_distractor_objects(b)

            self.task_instruction = self._generate_task_instruction()

            for i in range(5):
                self.scene.step()

    def _position_container_and_platform(self, b):
        container_xyz = torch.zeros((b, 3))
        container_xyz[:, 0] = 0.12
        container_xyz[:, 1] = 0.10
        container_xyz[:, 2] = 0.013
        self.container.set_pose(Pose.create_from_pq(container_xyz))
        
        platform_xyz = torch.zeros((b, 3))
        platform_xyz[:, 0] = 0.20
        platform_xyz[:, 1] = -0.10
        platform_xyz[:, 2] = 0.08
        self.platform.set_pose(Pose.create_from_pq(platform_xyz))

    def _setup_spatial_arrangement(self, b):
        container_pos = self.container.pose.p
        platform_pos = self.platform.pose.p
        target_obj = self.objects[self.target_object_color]
        intermediate_obj = self.objects[self.intermediate_object_color]
        
        self.stack_objects = set()
        
        if self.current_complex_relation == "on_top_of_inside":
            intermediate_xyz = torch.zeros((b, 3))
            intermediate_xyz[:, :2] = container_pos[:, :2]
            intermediate_xyz[:, 2] = container_pos[:, 2] + self.cube_half_size + 0.01
            intermediate_obj.set_pose(Pose.create_from_pq(intermediate_xyz))
            
            target_xyz = intermediate_xyz.clone()
            target_xyz[:, 2] += self.cube_half_size * 2.1
            target_obj.set_pose(Pose.create_from_pq(target_xyz))
            
            intermediate_obj.set_linear_velocity(torch.zeros((b, 3)))
            intermediate_obj.set_angular_velocity(torch.zeros((b, 3)))
            target_obj.set_linear_velocity(torch.zeros((b, 3)))
            target_obj.set_angular_velocity(torch.zeros((b, 3)))
            
            self.used_objects = {self.target_object_color, self.intermediate_object_color}
            
        elif self.current_complex_relation == "in_front_of_stacked_items":
            stack_xyz = torch.zeros((b, 3))
            stack_xyz[:, 0] = -0.
            stack_xyz[:, 1] = -0.05
            stack_xyz[:, 2] = self.cube_half_size + 0.02
            intermediate_obj.set_pose(Pose.create_from_pq(stack_xyz))
            
            remaining_objects = [k for k in self.objects.keys() 
                               if k not in {self.target_object_color, self.intermediate_object_color}]
            
            if remaining_objects:
                stack_top_color = remaining_objects[0]
                stack_top_obj = self.objects[stack_top_color]
                
                stack_top_xyz = stack_xyz.clone()
                stack_top_xyz[:, 2] += self.cube_half_size * 2.1
                stack_top_obj.set_pose(Pose.create_from_pq(stack_top_xyz))
                
                self.stack_objects = {self.intermediate_object_color, stack_top_color}
                
                target_xyz = stack_xyz.clone()
                target_xyz[:, 0] -= 0.08
                target_obj.set_pose(Pose.create_from_pq(target_xyz))
                
                self.used_objects = {self.target_object_color, self.intermediate_object_color, stack_top_color}
            else:
                self.stack_objects = {self.intermediate_object_color}
                target_xyz = stack_xyz.clone()
                target_xyz[:, 0] -= 0.08
                target_obj.set_pose(Pose.create_from_pq(target_xyz))
                
                self.used_objects = {self.target_object_color, self.intermediate_object_color}
                
        elif self.current_complex_relation == "inside_beside_platform":
            container_beside_xyz = torch.zeros((b, 3))
            container_beside_xyz[:, 0] = platform_pos[:, 0] - 0.16
            container_beside_xyz[:, 1] = platform_pos[:, 1]
            container_beside_xyz[:, 2] = 0.013
            self.container.set_pose(Pose.create_from_pq(container_beside_xyz))
            
            target_xyz = torch.zeros((b, 3))
            target_xyz[:, :2] = container_beside_xyz[:, :2]
            target_xyz[:, 2] = container_beside_xyz[:, 2] + self.cube_half_size + 0.01
            target_obj.set_pose(Pose.create_from_pq(target_xyz))
            
            target_obj.set_linear_velocity(torch.zeros((b, 3)))
            target_obj.set_angular_velocity(torch.zeros((b, 3)))
            
            intermediate_xyz = torch.zeros((b, 3))
            intermediate_xyz[:, 0] = 0.25
            intermediate_xyz[:, 1] = 0.15
            intermediate_xyz[:, 2] = self.cube_half_size
            intermediate_obj.set_pose(Pose.create_from_pq(intermediate_xyz))
            
            intermediate_obj.set_linear_velocity(torch.zeros((b, 3)))
            intermediate_obj.set_angular_velocity(torch.zeros((b, 3)))
            
            self.used_objects = {self.target_object_color, self.intermediate_object_color}
            
        elif self.current_complex_relation == "on_top_of_beside_container":
            intermediate_xyz = torch.zeros((b, 3))
            intermediate_xyz[:, 0] = container_pos[:, 0]
            intermediate_xyz[:, 1] = container_pos[:, 1] + 0.18
            intermediate_xyz[:, 2] = self.cube_half_size
            intermediate_obj.set_pose(Pose.create_from_pq(intermediate_xyz))
            
            target_xyz = intermediate_xyz.clone()
            target_xyz[:, 2] += self.cube_half_size * 2.1
            target_obj.set_pose(Pose.create_from_pq(target_xyz))
            
            intermediate_obj.set_linear_velocity(torch.zeros((b, 3)))
            intermediate_obj.set_angular_velocity(torch.zeros((b, 3)))
            target_obj.set_linear_velocity(torch.zeros((b, 3)))
            target_obj.set_angular_velocity(torch.zeros((b, 3)))
            
            self.used_objects = {self.target_object_color, self.intermediate_object_color}

    def _position_distractor_objects(self, b):
        remaining_objects = [k for k in self.objects.keys() if k not in self.used_objects]
        
        far_positions = [
            [0.30, 0.20],
            [0.30, -0.25],
        ]
        
        for i, color in enumerate(remaining_objects):
            obj_xyz = torch.zeros((b, 3))
            pos = far_positions[i % len(far_positions)]
            obj_xyz[:, :2] = torch.tensor(pos)
            obj_xyz[:, 2] = self.cube_half_size
            self.objects[color].set_pose(Pose.create_from_pq(obj_xyz))
            
            self.objects[color].set_linear_velocity(torch.zeros((b, 3)))
            self.objects[color].set_angular_velocity(torch.zeros((b, 3)))

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
            
            obs.update(
                target_object_pose=target_obj.pose.raw_pose,
                intermediate_object_pose=intermediate_obj.pose.raw_pose,
                container_pose=self.container.pose.raw_pose,
                platform_pose=self.platform.pose.raw_pose,
                tcp_to_target_pos=target_obj.pose.p - self.agent.tcp.pose.p,
                tcp_to_intermediate_pos=intermediate_obj.pose.p - self.agent.tcp.pose.p,
                is_grasping_target=self.agent.is_grasping(target_obj),
                is_grasping_intermediate=self.agent.is_grasping(intermediate_obj),
            )
            
            for color_name, obj in self.objects.items():
                obs[f"{color_name}_object_pose"] = obj.pose.raw_pose
                obs[f"is_grasping_{color_name}"] = self.agent.is_grasping(obj)
            
            obs.update(
                cube_half_size=torch.tensor(self.cube_half_size, device=self.device),
                lift_thresh=torch.tensor(self.lift_thresh, device=self.device),
            )
                
        return obs

    def _encode_complex_relation(self, relation):
        relation_map = {
            "on_top_of_inside": 0,
            "in_front_of_stacked_items": 1,
            "inside_beside_platform": 2,
            "on_top_of_beside_container": 3,
        }
        return relation_map.get(relation, 0)

    def _encode_color(self, color):
        color_map = {"red": 0, "green": 1, "blue": 2, "yellow": 3}
        return color_map.get(color, 0)

    def _encode_container_type(self, container_type):
        container_map = {"plate": 0}
        return container_map.get(container_type, 0)

    def _encode_task_type(self, task_type):
        task_map = {"spatial_hard": 0, "kitchen_assembly": 1, "spatial_medium": 2}
        return task_map.get(task_type, 0)

    def _generate_task_instruction(self):
        if self.current_complex_relation == "on_top_of_inside":
            return "Pick the cube that is on top of the cube inside the container"
        elif self.current_complex_relation == "in_front_of_stacked_items":
            return "Pick the cube that is in front of the two stacked cubes"
        elif self.current_complex_relation == "inside_beside_platform":
            return "Pick the cube inside the container that is beside the platform"
        elif self.current_complex_relation == "on_top_of_beside_container":
            return "Pick the cube on top of the cube that is beside the container"
        else:
            raise ValueError(f"Unknown spatial relation: {self.current_complex_relation}")

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
            "instruction": self.task_instruction
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        target_obj = self.objects[self.target_object_color]
        
        tcp_to_target_dist = torch.linalg.norm(
            target_obj.pose.p - self.agent.tcp_pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_target_dist)
        
        grasping_reward = self.agent.is_grasping(target_obj).float() * 2
        
        lift_height = target_obj.pose.p[:, 2] - self.cube_half_size
        lifting_reward = torch.clamp(lift_height / self.lift_thresh, 0, 1) * 2
        
        spatial_bonus = self._compute_spatial_reasoning_bonus()
        wrong_grasp_penalty = self._compute_wrong_grasp_penalty()
        
        reward = reaching_reward + grasping_reward + lifting_reward + spatial_bonus - wrong_grasp_penalty
        
        reward[info["success"]] += 5
        
        return reward

    def _compute_spatial_reasoning_bonus(self):
        if self.current_complex_relation == "on_top_of_inside":
            container_to_tcp_dist = torch.linalg.norm(
                self.container.pose.p[:, :2] - self.agent.tcp_pose.p[:, :2], axis=1
            )
            return (1 - torch.tanh(3 * container_to_tcp_dist)) * 0.5
        
        return torch.zeros_like(self.agent.tcp_pose.p[:, 0])

    def _compute_wrong_grasp_penalty(self):
        penalty = torch.zeros_like(self.agent.tcp_pose.p[:, 0])
        for color, obj in self.objects.items():
            if color != self.target_object_color:
                wrong_grasp = self.agent.is_grasping(obj).float()
                penalty += wrong_grasp * 0.5
        return penalty

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 10
    
    def get_task_instruction(self):
        return self.task_instruction