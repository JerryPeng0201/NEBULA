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

@register_env("Spatial-PickClosest-Medium", max_episode_steps=75)
class SpatialMediumPickClosestEnv(BaseEnv):

    """Task Description:
    Pick the cube closest to the reference cube (red cube). 
    The robot must calculate the distance between the reference cube and the target cubes, 
    and pick the closest one.

    Randomizations:
    - Target cubes are placed at varying distances from the reference cube.
    - Colors of the cubes are randomized.

    Success Conditions:
    - Correctly pick and lift the closest cube.
    """
    
    SUPPORTED_ROBOTS = ["panda", "fetch"]
    agent: Union[Panda, Fetch]
    
    cube_half_size = 0.02
    lift_thresh = 0.05
    task_instruction = "Pick the cube closest to the red cube."
    
    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.object_colors = {
            "red": [1, 0, 0, 1],
            "green": [0, 1, 0, 1], 
            "blue": [0, 0, 1, 1],
            "yellow": [1, 1, 0, 1],
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

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.table_scene.build()
        
        # Create reference cube (red cube) - make it kinematic to keep it fixed
        self.reference_cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=self.object_colors["red"],
            name="reference_cube",
            body_type="kinematic",  # Fixed position
            initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size])
        )
        
        # Create target cubes - these should be dynamic so they can be picked up
        self.target_cubes = {}
        colors = ["green", "blue", "yellow"]  # Exclude red as it's the reference
        distances = [0.1, 0.2, 0.3]  # Distances from the reference cube
        
        for i, color_name in enumerate(colors):
            position = [distances[i], 0, self.cube_half_size]  # Place along x-axis
            obj = actors.build_cube(
                self.scene,
                half_size=self.cube_half_size,
                color=self.object_colors[color_name],
                name=f"{color_name}_cube",
                body_type="dynamic",  # Make them dynamic so they can be picked up
                initial_pose=sapien.Pose(p=position)
            )
            self.target_cubes[color_name] = obj

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            
            # Randomize positions for target cubes
            distances = [0.1, 0.2, 0.3]
            random.shuffle(distances)
            
            colors = list(self.target_cubes.keys())
            for i, color_name in enumerate(colors):
                # Create random position within table bounds
                x = distances[i] + np.random.uniform(-0.05, 0.05)  # Add some randomness
                y = np.random.uniform(-0.1, 0.1)
                position = [x, y, self.cube_half_size]
                
                # Set the new pose for this cube
                pose = sapien.Pose(p=position)
                self.target_cubes[color_name].set_pose(pose)
            
            # Determine the closest cube after randomization
            ref_pos = np.array([0, 0, self.cube_half_size])  # Reference cube position
            min_distance = float("inf")
            closest_cube = None
            
            for color_name, cube in self.target_cubes.items():
                cube_pos = np.array(cube.pose.p).reshape(-1) 
                distance = np.linalg.norm(cube_pos[:2] - ref_pos[:2]) 
                if distance < min_distance:
                    min_distance = distance
                    closest_cube = color_name
            
            self.target_object = closest_cube
            # print(f"Target Object: {self.target_object} (Distance: {min_distance:.2f})")
            
            # Let physics settle before proceeding
            for _ in range(10):
                self.scene.step()

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            target_object_encoded=self._encode_target_object(self.target_object),
            task_type_encoded=self._encode_task_type("pick_closest"),
        )
        
        if "state" in self.obs_mode:
            target_obj = self.target_cubes[self.target_object]
            
            obs.update(
                reference_cube_pose=self.reference_cube.pose.raw_pose,
                target_cube_pose=target_obj.pose.raw_pose,
                tcp_to_target_pos=target_obj.pose.p - self.agent.tcp.pose.p,
                tcp_to_reference_pos=self.reference_cube.pose.p - self.agent.tcp.pose.p,
                target_to_reference_pos=self.reference_cube.pose.p - target_obj.pose.p,
                is_grasping_target=self.agent.is_grasping(target_obj),
            )
            
            # Add all cube poses and grasping states
            for color_name, obj in self.target_cubes.items():
                obs[f"{color_name}_cube_pose"] = obj.pose.raw_pose
                obs[f"is_grasping_{color_name}"] = self.agent.is_grasping(obj)
                obs[f"distance_{color_name}_to_reference"] = torch.linalg.norm(
                    obj.pose.p[:, :2] - self.reference_cube.pose.p[:, :2], axis=1
                )
                
        return obs

    def _encode_target_object(self, target_object):
        """Encode target object color as integer: green=0, blue=1, yellow=2"""
        target_map = {"green": 0, "blue": 1, "yellow": 2}
        return target_map.get(target_object, 0)

    def _encode_task_type(self, task_type):
        """Encode task type as integer: pick_closest=0, pick=1, place=2"""
        task_map = {"pick_closest": 0, "pick": 1, "place": 2}
        return task_map.get(task_type, 0)

    def evaluate(self):
        target_obj = self.target_cubes[self.target_object]
        
        is_lifted = target_obj.pose.p[:, 2] > self.cube_half_size + self.lift_thresh
        is_grasped = self.agent.is_grasping(target_obj)
        is_robot_static = self.agent.is_static(0.2)
        
        success = is_lifted & is_grasped & is_robot_static
        
        return {
            "success": success,
            "is_lifted": is_lifted, 
            "is_grasped": is_grasped,
            "is_robot_static": is_robot_static,
            "target_object": self.target_object,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        target_obj = self.target_cubes[self.target_object]
        
        # 1. Reaching reward
        tcp_to_target_dist = torch.linalg.norm(
            torch.tensor(target_obj.pose.p) - torch.tensor(self.agent.tcp_pose.p), axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_target_dist)
        
        # 2. Grasping reward
        grasping_reward = self.agent.is_grasping(target_obj).float() * 1.5
        
        # 3. Lifting reward
        lift_height = target_obj.pose.p[:, 2] - self.cube_half_size
        lifting_reward = torch.clamp(lift_height / self.lift_thresh, 0, 1) * 1.5
        
        # 4. Wrong object penalty
        wrong_grasp_penalty = self._compute_wrong_grasp_penalty()
        
        reward = reaching_reward + grasping_reward + lifting_reward - wrong_grasp_penalty
        
        # Success bonus
        reward[info["success"]] += 3
            
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        """Normalize the dense reward to a range of [0, 1]."""
        dense_reward = self.compute_dense_reward(obs=obs, action=action, info=info)
        max_reward = 7.0
        normalized_reward = dense_reward / max_reward
        return normalized_reward

    def _compute_wrong_grasp_penalty(self):
        """Penalty for grasping wrong objects"""
        penalty = torch.zeros_like(self.agent.tcp_pose.p[:, 0])
        for color, obj in self.target_cubes.items():
            if color != self.target_object:
                wrong_grasp = self.agent.is_grasping(obj).float()
                penalty += wrong_grasp * 0.3
        return penalty