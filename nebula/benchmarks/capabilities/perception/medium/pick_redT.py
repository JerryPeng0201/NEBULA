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


@register_env("Perception-PickRedT-Medium", max_episode_steps=50)
class PerceptionPickRedTMediumEnv(BaseEnv):
    """
    **Task Description:**
    Grasp the red T-shape among three objects of different shapes (blue sphere, green cube, and red T-shape).
    This is a perception task that tests the robot's ability to identify and grasp the 
    correct object based on its unique T-shape.

    **Randomizations:**
    - The objects are placed at fixed positions: blue sphere at y=-0.06, green cube at y=0.0, 
    and red T-shape at y=0.06, all at x=-0.08.

    **Success Conditions:**
    - The red T-shape is grasped by the robot. No placement is required for perception tasks.
    """

    SUPPORTED_ROBOTS = ["panda", "fetch"]

    # Specify some supported robot types
    agent: Union[Panda, Fetch]

    # set common size for all objects
    object_size = 0.02  # radius for sphere, half_size for cube
    
    # T-shape properties (bigger T)
    T_mass = 0.1
    T_dynamic_friction = 0.3
    T_static_friction = 0.3
    TARGET_RED = np.array([194, 19, 22, 255]) / 255

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

    def create_small_tee(self, name="small_tee", target=False, base_color=None):
        """Returns bigger 3D CAD of create_tee - scaled up for better grasping"""
        if base_color is None:
            base_color = self.TARGET_RED
        
        # scaled dimensions (bigger T for perception task)
        scale_factor = 0.6
        box1_half_w = (0.2 / 2) * scale_factor  # 0.035
        box1_half_h = (0.05 / 2) * scale_factor  # 0.00875
        half_thickness = (0.04 / 2) * scale_factor if not target else 1e-4  # 0.007

        # scaled center of mass calculation
        com_y = 0.0375 * scale_factor

        builder = self.scene.create_actor_builder()
        first_block_pose = sapien.Pose([0.0, 0.0 - com_y, 0.0])
        first_block_size = [box1_half_w, box1_half_h, half_thickness]
        
        if not target:
            builder._mass = self.T_mass
            tee_material = sapien.pysapien.physx.PhysxMaterial(
                static_friction=self.T_static_friction,
                dynamic_friction=self.T_dynamic_friction,
                restitution=0,
            )
            builder.add_box_collision(
                pose=first_block_pose,
                half_size=first_block_size,
                material=tee_material,
            )
        builder.add_box_visual(
            pose=first_block_pose,
            half_size=first_block_size,
            material=sapien.render.RenderMaterial(
                base_color=base_color,
            ),
        )

        # second block (vertical part)
        second_block_pose = sapien.Pose([0.0, 4 * (box1_half_h) - com_y, 0.0])
        second_block_size = [box1_half_h, (3 / 4) * (box1_half_w), half_thickness]
        if not target:
            builder.add_box_collision(
                pose=second_block_pose,
                half_size=second_block_size,
                material=tee_material,
            )
        builder.add_box_visual(
            pose=second_block_pose,
            half_size=second_block_size,
            material=sapien.render.RenderMaterial(
                base_color=base_color,
            ),
        )
        builder.initial_pose = sapien.Pose(p=[0, 0, 0.1])
        if not target:
            return builder.build_dynamic(name=name)
        else:
            return builder.build_kinematic(name=name)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        # load the table
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # load the blue sphere
        self.blue_sphere = actors.build_sphere(
            self.scene,
            radius=self.object_size,
            color=np.array([12, 42, 160, 255]) / 255,  # blue
            name="blue_sphere",
            body_type="dynamic",
            initial_pose=Pose.create_from_pq(p=[-0.08, -0.06, self.object_size], q=[1, 0, 0, 0])
        )
        
        # load the green cube
        self.green_cube = actors.build_cube(
            self.scene,
            half_size=self.object_size,
            color=np.array([0, 180, 0, 255]) / 255,  # green
            name="green_cube",
            body_type="dynamic",
            initial_pose=Pose.create_from_pq(p=[-0.08, 0.0, self.object_size], q=[1, 0, 0, 0])
        )

        # load the red T-shape (bigger now)
        self.red_tee = self.create_small_tee(name="red_tee", target=False)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            # init the table scene
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            xyz = torch.zeros((b, 3))
            q = [1, 0, 0, 0]
            
            # Blue sphere position (fixed)
            xyz[..., 0] = -0.08  # fixed x position
            xyz[..., 1] = -0.06  # fixed y position  
            xyz[..., 2] = self.object_size  # on the table
            blue_sphere_pose = Pose.create_from_pq(p=xyz.clone(), q=q)
            self.blue_sphere.set_pose(blue_sphere_pose)

            # Green cube position (fixed)
            xyz[..., 0] = -0.08  # same x as sphere
            xyz[..., 1] = 0.0    # center y position
            xyz[..., 2] = self.object_size  # on the table
            green_cube_pose = Pose.create_from_pq(p=xyz.clone(), q=q)
            self.green_cube.set_pose(green_cube_pose)

            # Red T-shape position (fixed) - bigger T now (scale_factor 0.6)
            xyz[..., 0] = -0.08  # same x as other objects
            xyz[..., 1] = 0.06   # fixed y position, spaced from others
            xyz[..., 2] = 0.012  # T thickness/2 (0.04/2 * 0.6 = 0.012)
            red_tee_pose = Pose.create_from_pq(p=xyz, q=q)
            self.red_tee.set_pose(red_tee_pose)

    def evaluate(self):
        # Check if objects are grasped
        is_blue_sphere_grasped = self.agent.is_grasping(self.blue_sphere)
        is_green_cube_grasped = self.agent.is_grasping(self.green_cube)
        is_red_tee_grasped = self.agent.is_grasping(self.red_tee)
        
        # Success: grasp the red T-shape - perception task only requires grasping the correct object
        success = is_red_tee_grasped
        
        return {
            "is_blue_sphere_grasped": is_blue_sphere_grasped,
            "is_green_cube_grasped": is_green_cube_grasped,
            "is_red_tee_grasped": is_red_tee_grasped,
            "success": success,
        }

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            is_blue_sphere_grasped=info["is_blue_sphere_grasped"],
            is_green_cube_grasped=info["is_green_cube_grasped"],
            is_red_tee_grasped=info["is_red_tee_grasped"],
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )
        if "state" in self.obs_mode:
            obs.update(
                blue_sphere_pose=self.blue_sphere.pose.raw_pose,
                green_cube_pose=self.green_cube.pose.raw_pose,
                red_tee_pose=self.red_tee.pose.raw_pose,
                tcp_to_blue_sphere_pos=self.blue_sphere.pose.p - self.agent.tcp.pose.p,
                tcp_to_green_cube_pos=self.green_cube.pose.p - self.agent.tcp.pose.p,
                tcp_to_red_tee_pos=self.red_tee.pose.p - self.agent.tcp.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # reaching reward - reward for getting close to the RED T-SHAPE (the target)
        tcp_pose = self.agent.tcp.pose.p
        red_tee_pos = self.red_tee.pose.p
        
        obj_to_tcp_dist_red_tee = torch.linalg.norm(tcp_pose - red_tee_pos, axis=1)
        reward = 2 * (1 - torch.tanh(5 * obj_to_tcp_dist_red_tee))

        # grasp reward - reward for grasping the correct (red T-shape) object
        is_red_tee_grasped = info["is_red_tee_grasped"]
        reward[is_red_tee_grasped] = 5.0

        # success reward - successfully grasped the red T-shape
        reward[info["success"]] = 10.0
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
        # this should be equal to compute_dense_reward / max possible reward
        max_reward = 10.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward