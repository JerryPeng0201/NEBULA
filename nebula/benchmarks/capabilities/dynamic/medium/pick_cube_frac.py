from typing import Any, Dict, Union

import numpy as np
import sapien
import torch

from nebula.core.embodiment.robots import Fetch, Panda
from nebula.core.simulation.engine import BaseEnv
from nebula.core.simulation.utils import randomization
from nebula.core.sensors.camera import CameraConfig
from nebula.utils import common, sapien_utils
from nebula.utils.building import actors
from nebula.utils.registration import register_env
from nebula.utils.scene_builder.table import TableSceneBuilder
from nebula.utils.structs.pose import Pose

@register_env("Dynamic-PickCubeWithSliding-Medium", max_episode_steps=300)
class SlidingPickCubeEnv(BaseEnv):
    """
    **Task Description:**
    A pick and place task on a sloped surface placed on a table where the cube slides 
    down due to gravity. The robot must catch and pick up the sliding cube while the 
    friction coefficient changes mid-episode, affecting the cube's sliding speed.

    **Key Features:**
    - Sloped ramp placed on a stable table
    - Cube continuously slides down the slope due to gravity
    - Dynamic friction coefficient changes mid-episode
    - Robot must adapt to changing cube velocity
    - Goal region at the top of the slope for extra challenge

    **Randomizations:**
    - Initial cube position on the slope
    - Slope angle (15-20 degrees)
    - Friction change timing and values
    - Goal region position

    Success Conditions:
    - Pick the correct colored cube that satisfies the spatial relation
    - Cube is lifted above minimum height (0.05m)
    - Robot is static after completion
    """

    SUPPORTED_ROBOTS = ["panda", "fetch", "panda_wristcam"]
    agent: Union[Panda, Fetch]

    def __init__(
        self,
        *args,
        robot_uids="panda",
        robot_init_qpos_noise=0.02,
        slope_angle_range=(15, 20),
        **kwargs
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.slope_angle_range = slope_angle_range
        
        self.cube_half_size = 0.02
        self.slope_length_range = (0.1, 0.3)
        self.slope_length = None
        self.slope_width = 0.4
        
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
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        # Build standard table first
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        
        # Build slope on top of table
        self._build_slope_on_table()
        
        # Create sliding cube
        self.cube_half_size_tensor = common.to_tensor([self.cube_half_size] * 3, device=self.device)
        self.cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[1, 0, 0, 1],  # Red cube
            name="sliding_cube",
            initial_pose=sapien.Pose(p=[0, 0, 0.1]),
        )

    def _build_slope_on_table(self):
        """Build a sloped ramp surface on the table"""
        builder = self.scene.create_actor_builder()
        
        friction_value = 0.2 + torch.rand(1).item() * 0.1

        self.slope_material = sapien.pysapien.physx.PhysxMaterial(
            static_friction=friction_value,
            dynamic_friction=friction_value,
            restitution=0.05
        )

        # Build slope collision
        self.slope_length = torch.rand(1).item() * (self.slope_length_range[1] - \
            self.slope_length_range[0]) + self.slope_length_range[0]
        slope_half_size = [self.slope_length/2, self.slope_width/2, 0.015]
        builder.add_box_collision(
            half_size=slope_half_size,
            material=self.slope_material,
            density=1000.0
        )
        
        # Build slope visual
        builder.add_box_visual(
            half_size=slope_half_size,
            material=sapien.render.RenderMaterial(base_color=[0.8, 0.6, 0.4, 1])
        )
        
        builder.set_physx_body_type("kinematic")
        builder.initial_pose = sapien.Pose(p=[0, 0, 0.04])
        
        self.slope = builder.build(name="slope_on_table")

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # Randomize slope angle
            slope_angles = torch.rand((b,), device=self.device) * (
                self.slope_angle_range[1] - self.slope_angle_range[0]
            ) + self.slope_angle_range[0]
            
            slope_angles_rad = torch.deg2rad(slope_angles)

            # Position slope
            slope_xyz = torch.zeros((b, 3))
            slope_xyz[:, 0] = 0.05
            slope_xyz[:, 1] = 0.0
            slope_xyz[:, 2] = 0.0
            
            slope_rotations = torch.zeros((b, 4))
            slope_rotations[:, 0] = torch.cos(slope_angles_rad / 2)
            slope_rotations[:, 2] = torch.sin(slope_angles_rad / 2)
            
            self.slope.set_pose(Pose.create_from_pq(p=slope_xyz, q=slope_rotations))

            # Position cube
            cube_xyz = torch.zeros((b, 3))
            for i in range(b):
                angle = slope_angles_rad[i]
                
                x_pos = -self.slope_length/2 * 0.8
                
                slope_center_x = slope_xyz[i, 0]
                world_x = slope_center_x + x_pos * torch.cos(angle)
                world_z = slope_xyz[i, 2] + abs(x_pos) * torch.sin(angle) + self.cube_half_size + 0.02 
                
                cube_xyz[i, 0] = world_x
                cube_xyz[i, 1] = torch.rand(1) * 0.4 - 0.2
                cube_xyz[i, 2] = world_z

            self.cube.set_pose(Pose.create_from_pq(p=cube_xyz, q=[1, 0, 0, 0]))
            self.slope_angles = slope_angles

    def _monitor_cube_bounds(self):
        """Monitor cube position and reset if it falls off the table/slope"""
        cube_pos = self.cube.pose.p
        
        out_of_bounds = (
            (cube_pos[:, 2] < -0.05) | 
            (torch.abs(cube_pos[:, 1]) > 0.35) |
            (cube_pos[:, 0] < -0.35) | 
            (cube_pos[:, 0] > 0.35) 
        )
        
        if out_of_bounds.any():
            # Reset cube to a better position on the slope
            reset_pos = cube_pos.clone()
            
            for i in range(len(reset_pos)):
                if out_of_bounds[i]:
                    angle = self.friction_state['slope_angles'][i] * np.pi / 180
                    slope_center_x = 0.05
                    
                    x_pos = self.slope_length/2 * 0.4 
                    world_x = slope_center_x + x_pos * np.cos(angle)
                    world_z = 0.04 + x_pos * np.sin(angle) + self.cube_half_size + 0.03
                    
                    reset_pos[i, 0] = world_x
                    reset_pos[i, 1] = 0.0
                    reset_pos[i, 2] = world_z
            
            # Reset velocity
            zero_vel = torch.zeros_like(self.cube.linear_velocity)
            zero_angvel = torch.zeros_like(self.cube.angular_velocity)
            self.cube.set_linear_velocity(zero_vel)
            self.cube.set_angular_velocity(zero_angvel)
            self.cube.set_pose(Pose.create_from_pq(p=reset_pos, q=[1, 0, 0, 0]))

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self._monitor_cube_bounds()
        return obs, reward, terminated, truncated, info

    def _monitor_cube_bounds(self):
        """Monitor cube position and reset if it falls off the table/slope"""
        cube_pos = self.cube.pose.p
        
        # Check if cube fell off the table or went too far
        out_of_bounds = (
            (cube_pos[:, 2] < -0.1) |  # Fell below ground
            (torch.abs(cube_pos[:, 1]) > 0.4) |  # Fell off table sides
            (cube_pos[:, 0] < -0.4) |  # Went too far back
            (cube_pos[:, 0] > 0.4)     # Went too far forward
        )
        
        if out_of_bounds.any():
            # Reset cube to a position on the slope
            reset_pos = cube_pos.clone()
            reset_pos[out_of_bounds, 0] = 0.15  # Middle of slope
            reset_pos[out_of_bounds, 1] = 0.0   # Center of slope
            reset_pos[out_of_bounds, 2] = 0.15  # Above slope surface
            
            # Reset velocity
            zero_vel = torch.zeros_like(self.cube.linear_velocity)
            zero_angvel = torch.zeros_like(self.cube.angular_velocity)
            self.cube.set_linear_velocity(zero_vel)
            self.cube.set_angular_velocity(zero_angvel)
            self.cube.set_pose(Pose.create_from_pq(p=reset_pos, q=[1, 0, 0, 0]))

    def evaluate(self):
        cube_pos = self.cube.pose.p
        
        is_cube_grasped = self.agent.is_grasping(self.cube)
        cube_height = cube_pos[:, 2] - self.cube_half_size
        is_cube_lifted = cube_height > 0.05
        
        is_robot_static = self.agent.is_static(0.2)
        cube_velocity = torch.linalg.norm(self.cube.linear_velocity, axis=1)
        is_cube_stable = cube_velocity < 0.1
        
        success = is_cube_grasped & is_cube_lifted & is_robot_static & is_cube_stable
        
        return {
            "is_cube_grasped": is_cube_grasped,
            "is_cube_lifted": is_cube_lifted,
            "is_robot_static": is_robot_static,
            "is_cube_stable": is_cube_stable,
            "cube_height": cube_height,
            "cube_velocity": cube_velocity,
            "slope_angle": self.slope_angles,
            "success": success.bool(),
        }

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            cube_velocity=self.cube.linear_velocity,
            slope_angle=self.slope_angles,
        )
        
        if "state" in self.obs_mode:
            obs.update(
                cube_pose=self.cube.pose.raw_pose,
                tcp_to_cube_pos=self.cube.pose.p - self.agent.tcp.pose.p,
                slope_pose=self.slope.pose.raw_pose,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_pos = self.agent.tcp.pose.p
        cube_pos = self.cube.pose.p
        tcp_to_cube_dist = torch.linalg.norm(tcp_pos - cube_pos, axis=1)
        reaching_reward = 1 - torch.tanh(5.0 * tcp_to_cube_dist)
        grasping_reward = 3.0 * info["is_cube_grasped"].float()
        lifting_reward = 2.0 * torch.clamp(info["cube_height"] / 0.05, 0, 1)        
        success_reward = 10.0 * info["success"].float()
        total_reward = reaching_reward + grasping_reward + lifting_reward + success_reward
        return total_reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 17.0