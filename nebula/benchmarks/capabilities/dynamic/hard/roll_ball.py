from typing import Any, Dict, Union
import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

import nebula.core.simulation.utils.randomization as randomization
from nebula.core.embodiment.robots import Fetch, Panda
from nebula.core.simulation.engine import BaseEnv
from nebula.core.sensors.camera import CameraConfig
from nebula.utils import sapien_utils
from nebula.utils.building import actors
from nebula.utils.registration import register_env
from nebula.utils.scene_builder.table import TableSceneBuilder
from nebula.utils.structs.pose import Pose
from nebula.utils.structs.types import Array, GPUMemoryConfig, SimConfig

DYNAMIC_EASY_DOC_STRING = """Task Description:
Roll a ball to a target position while avoiding a bouncing ball that crosses the workspace.
This task tests the robot's ability to maintain task performance while dynamically avoiding obstacles.

Core Task:
- Roll a ball from its initial position to a red target region (same as RollBall-v1)
- Ball must be rolled (not picked up) to the target
- Success requires ball to be within target region and robot to be static

Dynamic Bouncing Ball:
- A bouncing ball crosses the workspace from left to right (or right to left)
- Ball bounces realistically with gravity and energy loss
- Initial velocity, angle, and timing are randomized
- Robot must avoid collision while maintaining task progress
- Tests dynamic obstacle avoidance and trajectory planning

Randomizations:
- Main ball initial position randomized on table
- Target position randomized on table
- Bouncing ball initial velocity (3-5 m/s horizontal)
- Bouncing ball launch angle (15-45 degrees)
- Bouncing ball start time (0-2 seconds delay)
- Bouncing ball direction (left-to-right or right-to-left)

Success Conditions:
- Main ball center is within target region (radius 0.1m)
- Main ball is on the table (not lifted)
- Robot is static after completion
- Task completed despite bouncing ball interference
"""

@register_env("Dynamic-RollBall-Hard", max_episode_steps=200)
class DynamicHardRollBallEnv(BaseEnv):
    
    SUPPORTED_ROBOTS = ["panda", "fetch"]
    agent: Union[Panda, Fetch]
    
    # Main ball and target parameters (same as RollBall-v1)
    goal_radius: float = 0.1
    ball_radius: float = 0.035
    
    # Bouncing ball parameters
    bouncing_ball_radius: float = 0.03
    bounce_restitution: float = 0.6  # Energy retained after bounce
    bounce_friction: float = 0.2
    
    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        
        # Bouncing ball state
        self.bounce_launched = False
        self.bounce_launch_time = 0
        self.time_since_reset = 0
        
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25, 
                max_rigid_patch_count=2**18
            ),
            spacing=20,
            sim_freq=120,
            control_freq=20
        )

    @property
    def _default_sensor_configs(self):
        # Base camera configuration
        base_pose = sapien_utils.look_at(eye=[0.7, 0, 0.5], target=[-0.1, 0, 0])
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
        back_right_pose = sapien_utils.look_at(eye=[-1, 1, 0.3], target=[0, 0, 0.1])
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
        back_left_pose = sapien_utils.look_at(eye=[-1, 1, 0.3], target=[0, 0, 0.1])
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
        front_right_pose = sapien_utils.look_at(eye=[1, -1, 0.3], target=[0, 0, 0.1])
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
        front_left_pose = sapien_utils.look_at(eye=[1, 1, 0.3], target=[0, 0, 0.1])
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
        pose = sapien_utils.look_at([-0.6, 1.3, 0.8], [0.0, 0.13, 0.0])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        
        # Main ball (blue, same as RollBall-v1)
        self.ball = actors.build_sphere(
            self.scene,
            radius=self.ball_radius,
            color=[0, 0.2, 0.8, 1],
            name="ball",
            body_type="dynamic",
            initial_pose=sapien.Pose(p=[0, 0, 0.1]),
            
        )
        
        # Goal region (same as RollBall-v1)
        self.goal_region = actors.build_red_white_target(
            self.scene,
            radius=self.goal_radius,
            thickness=1e-5,
            name="goal_region",
            add_collision=False,
            body_type="kinematic",
            initial_pose=sapien.Pose(p=[0, 0, 0.1]),
        )
        
        # Bouncing ball (yellow/orange for visibility)
        self.bouncing_ball = actors.build_sphere(
            self.scene,
            radius=self.bouncing_ball_radius,
            color=[1.0, 0.7, 0.0, 1],
            name="bouncing_ball",
            body_type="dynamic",
            initial_pose=sapien.Pose(p=[0, 0, -1.0]),  # Start below table
        )
        
        # Set bouncing ball physical properties
        self._set_bouncing_ball_material()
        
        # Track if main ball reached pushing position
        self.reached_status = torch.zeros(self.num_envs, dtype=torch.float32)

    def _set_bouncing_ball_material(self):
        """Set physical material for realistic bouncing"""
        # In NEBULA, actors have collision_shapes property
        if hasattr(self.bouncing_ball, 'collision_shapes'):
            for collision_shape in self.bouncing_ball.collision_shapes:
                material = collision_shape.physical_material
                material.static_friction = self.bounce_friction
                material.dynamic_friction = self.bounce_friction
                material.restitution = self.bounce_restitution

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self.reached_status = self.reached_status.to(self.device)
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            
            # Reset robot pose (same as RollBall-v1)
            robot_pose = Pose.create_from_pq(
                p=[-0.1, 1.0, 0], q=[0.7071, 0, 0, -0.7072]
            )
            self.agent.robot.set_pose(robot_pose)
            
            # Initialize main ball position (same as RollBall-v1)
            xyz = torch.zeros((b, 3))
            xyz[..., 0] = (torch.rand((b)) * 2 - 1) * 0.3 - 0.1
            xyz[..., 1] = torch.rand((b)) * 0.2 + 0.3
            xyz[..., 2] = self.ball_radius
            self.ball.set_pose(Pose.create_from_pq(p=xyz, q=[1, 0, 0, 0]))
            
            # Initialize goal position (same as RollBall-v1)
            xyz_goal = torch.zeros((b, 3))
            xyz_goal[..., 0] = (torch.rand((b)) * 2 - 1) * 0.3 - 0.1
            xyz_goal[..., 1] = torch.rand((b)) * 0.2 - 1.0 + self.goal_radius
            xyz_goal[..., 2] = 1e-3
            self.goal_region.set_pose(
                Pose.create_from_pq(
                    p=xyz_goal,
                    q=euler2quat(0, np.pi / 2, 0),
                )
            )
            
            # Initialize bouncing ball parameters
            self._reset_bouncing_ball(b)
            
            # Reset status
            self.reached_status[env_idx] = 0.0
            self.time_since_reset = 0

    def _reset_bouncing_ball(self, b):
        """Reset bouncing ball for new episode"""
        # Hide bouncing ball initially (place below table)
        initial_pos = torch.zeros((b, 3))
        initial_pos[..., 0] = 0
        initial_pos[..., 1] = 0
        initial_pos[..., 2] = -1.0
        self.bouncing_ball.set_pose(Pose.create_from_pq(p=initial_pos))
        self.bouncing_ball.set_linear_velocity(torch.zeros((b, 3)))
        self.bouncing_ball.set_angular_velocity(torch.zeros((b, 3)))
        
        # Randomize launch parameters
        self.bounce_launched = False
        self.bounce_launch_time = np.random.uniform(0.5, 2.0)  # Launch after 0.5-2 seconds
        
        # Randomize launch direction (left to right or right to left)
        self.bounce_direction = np.random.choice([-1, 1])
        
        # Randomize initial velocity
        self.bounce_initial_speed = np.random.uniform(1.0, 2.0)  # m/s horizontal
        self.bounce_launch_angle = np.random.uniform(30, 60)  # degrees upward
        
        # Starting position (left or right side of table)
        self.bounce_start_x = 0.6 * self.bounce_direction
        self.bounce_start_y = np.random.uniform(0.1, 0.7)  # Along the table length
        self.bounce_start_z = self.bouncing_ball_radius + 0.2  # Start slightly above table

    def _launch_bouncing_ball(self):
        """Launch the bouncing ball across the workspace"""
        if self.bounce_launched:
            return
            
        b = self.num_envs
        
        # Set starting position
        launch_pos = torch.zeros((b, 3), device=self.device)
        launch_pos[..., 0] = self.bounce_start_x
        launch_pos[..., 1] = self.bounce_start_y
        launch_pos[..., 2] = self.bounce_start_z
        self.bouncing_ball.set_pose(Pose.create_from_pq(p=launch_pos))
        
        # Calculate initial velocity
        angle_rad = np.radians(self.bounce_launch_angle)
        vx = -self.bounce_direction * self.bounce_initial_speed * np.cos(angle_rad)
        vz = self.bounce_initial_speed * np.sin(angle_rad)
        vy = np.random.uniform(-0.5, 0.5)  # Small random y velocity
        
        launch_vel = torch.zeros((b, 3), device=self.device)
        launch_vel[..., 0] = vx
        launch_vel[..., 1] = vy
        launch_vel[..., 2] = vz
        self.bouncing_ball.set_linear_velocity(launch_vel)
        
        # Add small random angular velocity for realistic motion
        angular_vel = torch.randn((b, 3), device=self.device) * 2.0
        self.bouncing_ball.set_angular_velocity(angular_vel)
        
        self.bounce_launched = True

    def _check_bouncing_ball_reset(self):
        """Check if bouncing ball needs to be reset (went out of bounds)"""
        ball_pos = self.bouncing_ball.pose.p
        
        # Check if ball is out of bounds
        out_of_bounds = (
            (torch.abs(ball_pos[..., 0]) > 1.0) |  # Too far in x
            (torch.abs(ball_pos[..., 1]) > 1.5) |  # Too far in y
            (ball_pos[..., 2] < -0.5)              # Fell too far
        )
        
        if torch.any(out_of_bounds):
            # Hide the ball again
            hidden_pos = ball_pos.clone()
            hidden_pos[out_of_bounds, 2] = -1.0
            self.bouncing_ball.set_pose(Pose.create_from_pq(p=hidden_pos))
            
            # Stop its motion
            zero_vel = torch.zeros_like(self.bouncing_ball.linear_velocity)
            self.bouncing_ball.set_linear_velocity(
                torch.where(out_of_bounds.unsqueeze(-1), zero_vel, self.bouncing_ball.linear_velocity)
            )

    def step(self, action: torch.Tensor):
        """Override step to handle bouncing ball dynamics"""
        # Update time
        self.time_since_reset += 1.0 / self.control_freq
        
        # Launch bouncing ball if it's time
        if not self.bounce_launched and self.time_since_reset >= self.bounce_launch_time:
            self._launch_bouncing_ball()
        
        # Check if bouncing ball needs reset
        if self.bounce_launched:
            self._check_bouncing_ball_reset()
        
        # Call parent step
        obs, reward, terminated, truncated, info = super().step(action)
        
        return obs, reward, terminated, truncated, info

    def evaluate(self):
        is_obj_placed = (
            torch.linalg.norm(
                self.ball.pose.p[..., :2] - self.goal_region.pose.p[..., :2], axis=1
            )
            < self.goal_radius
        )
        
        return {
            "success": is_obj_placed,
        }

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )
        
        if self.obs_mode_struct.use_state:
            obs.update(
                goal_pos=self.goal_region.pose.p,
                ball_pose=self.ball.pose.raw_pose,
                ball_vel=self.ball.linear_velocity,
                tcp_to_ball_pos=self.ball.pose.p - self.agent.tcp.pose.p,
                ball_to_goal_pos=self.goal_region.pose.p - self.ball.pose.p,
                # Bouncing ball information
                bouncing_ball_pose=self.bouncing_ball.pose.raw_pose,
                bouncing_ball_vel=self.bouncing_ball.linear_velocity,
                bouncing_ball_active=torch.tensor([self.bounce_launched], dtype=torch.float32),
                time_until_launch=torch.tensor([max(0, self.bounce_launch_time - self.time_since_reset)], dtype=torch.float32),
            )
        
        return obs

    def compute_dense_reward(self, obs: Any, action: Array, info: Dict):
        # Base reward from RollBall-v1
        unit_vec = self.ball.pose.p - self.goal_region.pose.p
        unit_vec = unit_vec / torch.linalg.norm(unit_vec, axis=1, keepdim=True)
        tcp_hit_pose = Pose.create_from_pq(
            p=self.ball.pose.p + unit_vec * (self.ball_radius + 0.05),
        )
        tcp_to_hit_pose = tcp_hit_pose.p - self.agent.tcp.pose.p
        tcp_to_hit_pose_dist = torch.linalg.norm(tcp_to_hit_pose, axis=1)
        self.reached_status[tcp_to_hit_pose_dist < 0.04] = 1.0
        reaching_reward = 1 - torch.tanh(2 * tcp_to_hit_pose_dist)
        
        obj_to_goal_dist = torch.linalg.norm(
            self.ball.pose.p[..., :2] - self.goal_region.pose.p[..., :2], axis=1
        )
        reached_reward = 1 - torch.tanh(obj_to_goal_dist)
        
        # Base reward
        reward = (
            20 * reached_reward * self.reached_status
            + reaching_reward * (1 - self.reached_status)
            + self.reached_status
        )
        
        # Additional rewards for dynamic obstacle avoidance
        if self.bounce_launched:
            # Reward for maintaining safe distance from bouncing ball
            bouncing_dist = torch.linalg.norm(
                self.agent.tcp.pose.p - self.bouncing_ball.pose.p, axis=1
            )
            safety_reward = torch.tanh(bouncing_dist * 2)  # Reward being far from bouncing ball
            
            # Penalty for collision with bouncing ball
            collision_penalty = torch.where(bouncing_dist < 0.1, -5.0, 0.0)
            
            # Reward for continuing task despite distraction
            persistence_reward = reached_reward * 0.5
            
            reward = reward + safety_reward * 0.5 + collision_penalty + persistence_reward
        
        # Success bonus
        reward[info["success"]] = 30.0
        
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
        max_reward = 30.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward

DynamicHardRollBallEnv.__doc__ = DYNAMIC_EASY_DOC_STRING