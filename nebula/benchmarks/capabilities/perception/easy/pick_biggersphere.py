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


@register_env("Perception-PickBiggerSphere-Easy", max_episode_steps=50)
class PerceptionPickBiggerSphereEasyEnv(BaseEnv):
    """
    **Task Description:**
    Grasp the bigger sphere (large blue sphere) among two blue spheres of different sizes.
    This is a perception task that tests the robot's ability to identify and grasp the larger 
    of two similarly colored objects.

    **Randomizations:**
    - The spheres are placed at fixed positions: the large sphere at x=-0.08, y=-0.05 and 
    the small sphere at x=-0.08, y=0.05, ensuring a consistent distance of 0.1 units between them.

    **Success Conditions:**
    - The large blue sphere is grasped by the robot. No placement is required for perception tasks.
    """

    SUPPORTED_ROBOTS = ["panda", "fetch"]

    # Specify some supported robot types
    agent: Union[Panda, Fetch]

    # Sphere dimensions
    radius_large = 0.02  # radius of the large sphere
    radius_small = 0.012  # radius of the small sphere

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

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        # load the table
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # load the two blue spheres - one large and one small
        self.large_blue_sphere = actors.build_sphere(
            self.scene,
            radius=self.radius_large,
            color=np.array([12, 42, 160, 255]) / 255,
            name="large_blue_sphere",
            body_type="dynamic",
            initial_pose=Pose.create_from_pq(q=[1,0,0,0], p=[0,0,0])
        )
        
        self.small_blue_sphere = actors.build_sphere(
            self.scene,
            radius=self.radius_small,
            color=np.array([12, 42, 160, 255]) / 255,
            name="small_blue_sphere",
            body_type="dynamic",
            initial_pose=Pose.create_from_pq(q=[1,0,0,0], p=[0,0,0])
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            # init the table scene
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            xyz = torch.zeros((b, 3))
            q = [1, 0, 0, 0]
            
            # Large blue sphere position (fixed)
            xyz[..., 0] = -0.08  # fixed x position
            xyz[..., 1] = -0.05  # fixed y position  
            xyz[..., 2] = self.radius_large  # on the table
            large_blue_sphere_pose = Pose.create_from_pq(p=xyz.clone(), q=q)
            self.large_blue_sphere.set_pose(large_blue_sphere_pose)

            # Small blue sphere position (fixed, with guaranteed distance)
            xyz[..., 0] = -0.08  # same x as large sphere
            xyz[..., 1] = 0.05   # fixed y position, 0.1 units away from large sphere
            xyz[..., 2] = self.radius_small  # on the table
            small_blue_sphere_pose = Pose.create_from_pq(p=xyz, q=q)
            self.small_blue_sphere.set_pose(small_blue_sphere_pose)

    def evaluate(self):
        # Check if spheres are grasped
        is_large_sphere_grasped = self.agent.is_grasping(self.large_blue_sphere)
        is_small_sphere_grasped = self.agent.is_grasping(self.small_blue_sphere)
        
        # Success: grasp the bigger (large) sphere - perception task only requires grasping
        success = is_large_sphere_grasped
        
        return {
            "is_large_sphere_grasped": is_large_sphere_grasped,
            "is_small_sphere_grasped": is_small_sphere_grasped,
            "success": success,
        }

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            is_large_sphere_grasped=info["is_large_sphere_grasped"],
            is_small_sphere_grasped=info["is_small_sphere_grasped"],
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )
        if "state" in self.obs_mode:
            obs.update(
                large_sphere_pose=self.large_blue_sphere.pose.raw_pose,
                small_sphere_pose=self.small_blue_sphere.pose.raw_pose,
                tcp_to_large_sphere_pos=self.large_blue_sphere.pose.p - self.agent.tcp.pose.p,
                tcp_to_small_sphere_pos=self.small_blue_sphere.pose.p - self.agent.tcp.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # reaching reward - reward for getting close to the LARGE sphere (the target)
        tcp_pose = self.agent.tcp.pose.p
        large_sphere_pos = self.large_blue_sphere.pose.p
        
        obj_to_tcp_dist_large = torch.linalg.norm(tcp_pose - large_sphere_pos, axis=1)
        reward = 2 * (1 - torch.tanh(5 * obj_to_tcp_dist_large))

        # grasp reward - reward for grasping the correct (large) sphere
        is_large_grasped = info["is_large_sphere_grasped"]
        reward[is_large_grasped] = 5.0

        # success reward - successfully grasped the bigger sphere
        reward[info["success"]] = 10.0
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
        # this should be equal to compute_dense_reward / max possible reward
        max_reward = 10.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward