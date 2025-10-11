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


@register_env("Perception-PickRedSphere-Easy", max_episode_steps=50)
class PerceptionPickRedSphereEasyEnv(BaseEnv):
    """
    **Task Description:**
    Grasp the red sphere among two spheres of different colors (blue and red).
    This is a perception task that tests the robot's ability to identify and grasp the 
    correct object based on color.

    **Randomizations:**
    - The spheres are placed at randomized positions: blue sphere in x ∈ [-0.1, -0.05], y ∈ [-0.1, 0],
    and red sphere in x ∈ [-0.1, -0.05], y ∈ [0, 0.1].

    **Success Conditions:**
    - The red sphere is grasped by the robot. No placement is required for perception tasks.
    """

    SUPPORTED_ROBOTS = ["panda", "fetch"]

    # Specify some supported robot types
    agent: Union[Panda, Fetch]

    # Sphere dimensions
    radius = 0.02  # radius of the sphere

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

        # load the two spheres - blue and red
        self.blue_sphere = actors.build_sphere(
            self.scene,
            radius=self.radius,
            color=np.array([12, 42, 160, 255]) / 255,
            name="blue_sphere",
            body_type="dynamic",
            initial_pose=Pose.create_from_pq(p=[0, 0, self.radius], q=[1, 0, 0, 0])
        )
        
        self.red_sphere = actors.build_sphere(
            self.scene,
            radius=self.radius,
            color=np.array([255, 0, 0, 255]) / 255,
            name="red_sphere",
            body_type="dynamic",
            initial_pose=Pose.create_from_pq(p=[0, 0, self.radius], q=[1, 0, 0, 0])
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            # init the table scene
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # init the spheres with randomized positions
            xyz = torch.zeros((b, 3))
            xyz[..., 2] = self.radius  # on the table
            q = [1, 0, 0, 0]
            
            # Blue sphere position
            xyz[..., 0] = (torch.rand((b, 1)) * 0.05 - 0.1)[..., 0]  # x ∈ [-0.1, -0.05]
            xyz[..., 1] = (torch.rand((b, 1)) * 0.1 - 0.1)[..., 0]   # y ∈ [-0.1, 0]
            blue_sphere_pose = Pose.create_from_pq(p=xyz.clone(), q=q)
            self.blue_sphere.set_pose(blue_sphere_pose)

            # Red sphere position
            xyz[..., 0] = (torch.rand((b, 1)) * 0.05 - 0.1)[..., 0]  # x ∈ [-0.1, -0.05]
            xyz[..., 1] = (torch.rand((b, 1)) * 0.1)[..., 0]         # y ∈ [0, 0.1]
            red_sphere_pose = Pose.create_from_pq(p=xyz, q=q)
            self.red_sphere.set_pose(red_sphere_pose)

    def evaluate(self):
        # Check if spheres are grasped
        is_blue_sphere_grasped = self.agent.is_grasping(self.blue_sphere)
        is_red_sphere_grasped = self.agent.is_grasping(self.red_sphere)
        
        # Success: grasp the red sphere - perception task only requires grasping the correct object
        success = is_red_sphere_grasped
        
        return {
            "is_blue_sphere_grasped": is_blue_sphere_grasped,
            "is_red_sphere_grasped": is_red_sphere_grasped,
            "success": success,
        }

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            is_blue_sphere_grasped=info["is_blue_sphere_grasped"],
            is_red_sphere_grasped=info["is_red_sphere_grasped"],
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )
        if "state" in self.obs_mode:
            obs.update(
                blue_sphere_pose=self.blue_sphere.pose.raw_pose,
                red_sphere_pose=self.red_sphere.pose.raw_pose,
                tcp_to_blue_sphere_pos=self.blue_sphere.pose.p - self.agent.tcp.pose.p,
                tcp_to_red_sphere_pos=self.red_sphere.pose.p - self.agent.tcp.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # reaching reward - reward for getting close to the RED sphere (the target)
        tcp_pose = self.agent.tcp.pose.p
        red_sphere_pos = self.red_sphere.pose.p
        
        obj_to_tcp_dist_red = torch.linalg.norm(tcp_pose - red_sphere_pos, axis=1)
        reward = 2 * (1 - torch.tanh(5 * obj_to_tcp_dist_red))

        # grasp reward - reward for grasping the correct (red) sphere
        is_red_grasped = info["is_red_sphere_grasped"]
        reward[is_red_grasped] = 5.0

        # success reward - successfully grasped the red sphere
        reward[info["success"]] = 10.0
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
        # this should be equal to compute_dense_reward / max possible reward
        max_reward = 10.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward