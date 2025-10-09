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
from sapien.physx import PhysxMaterial

DISTRACTOR_BALL_DOC_STRING = """Task Description:
Pick up the red cube while being distracted by a moving yellow ball that rolls across the workspace.
The ball serves as a visual and dynamic distraction, testing the robot's ability to focus on the target object.

Task Components:
- Target: Red cube that needs to be picked up
- Distractor: yellow ball that rolls across the table during the task
- Environment: Table setup with clear workspace

Distractor Mechanics:
- yellow ball starts from a random edge of the table
- Ball rolls at moderate speed across the workspace
- Ball movement is timed to coincide with typical grasping attempts
- Ball does not interfere physically but provides visual distraction

Success Conditions:
- Pick the red cube (not the ball)
- Cube is lifted above minimum height (0.05m)
- Robot maintains static position after completion
- Task completion despite ball movement

Randomizations:
- Red cube position randomized on table
- Ball starting position and direction randomized
- Ball rolling speed slightly randomized
- Timing of ball movement varies
"""

@register_env("Dynamic-RollBallWithDistraction-Hard", max_episode_steps=100)
class DistractorBallPickCubeEnv(BaseEnv):
    
    SUPPORTED_ROBOTS = ["panda", "fetch"]
    agent: Union[Panda, Fetch]
    
    cube_half_size = 0.025
    ball_radius = 0.015
    lift_thresh = 0.05
    
    # Table workspace bounds
    TABLE_X_RANGE = 0.4  # [-0.2, 0.2] from robot's perspective
    TABLE_Y_RANGE = 0.6  # [-0.3, 0.3] from robot's perspective
    
    # Ball movement parameters
    BALL_SPEED_RANGE = [0.5, 0.8]  # m/s, moderate rolling speed
    BALL_START_DELAY_RANGE = [0, 5]  # Steps before ball starts moving
    
    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        
        # Fixed colors for clarity
        self.cube_color = [0.8, 0.1, 0.1, 1]      # Red cube (target)
        self.ball_color = [1.0, 0.6, 0.0, 1]      # yellow ball (distractor)
        
        # Ball movement state
        self.ball_start_step = 0
        self.ball_velocity = torch.zeros(3)
        self.ball_is_moving = False
        
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property 
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.4, 0, 0.3], target=[0, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[0.6, 0.6, 0.8], target=[0, 0, 0.1])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.table_scene.build()
        
        # Create red cube (target)
        self.target_cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=self.cube_color,
            name="red_cube",
            initial_pose=sapien.Pose(p=[0, 0, 1.0])  # Start high to avoid collision
        )
        
        self.distractor_ball = actors.build_sphere(
            self.scene,
            radius=self.ball_radius,
            color=self.ball_color,
            name="yellow_ball",
            body_type="dynamic",
            initial_pose=sapien.Pose(p=[0, 0, 1.0])  # Start high to avoid collision
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

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            
            # Reset ball movement state
            self.ball_is_moving = False
            self.current_step = 0
            
            # Randomize ball start delay
            self.ball_start_step = random.randint(*self.BALL_START_DELAY_RANGE)
            
            # Position target cube randomly on table
            self._position_target_cube(b)
            
            # Position ball at starting edge
            self._position_and_setup_ball(b)

    def _position_target_cube(self, b):
        """Position the red target cube randomly on the table"""
        cube_xyz = torch.zeros((b, 3))
        
        # Random position within table bounds, avoiding edges where ball will roll
        margin = 0.08  # Keep cube away from edges
        cube_xyz[:, 0] = (torch.rand(b) * 2 - 1) * (self.TABLE_X_RANGE / 2 - margin)
        cube_xyz[:, 1] = (torch.rand(b) * 2 - 1) * (self.TABLE_Y_RANGE / 2 - margin)
        cube_xyz[:, 2] = self.cube_half_size
        
        self.target_cube.set_pose(Pose.create_from_pq(cube_xyz))
        self.cube_initial_pos = cube_xyz[0].clone()

    def _position_and_setup_ball(self, b):
        """Position ball at starting edge and set up its rolling trajectory"""
        ball_xyz = torch.zeros((b, 3))
        
        # Choose random edge to start from
        edge_choice = random.randint(0, 3)  # 0: left, 1: right, 2: front, 3: back
        
        if edge_choice == 0:  # Left edge
            ball_xyz[:, 0] = (torch.rand(b) * 2 - 1) * self.TABLE_X_RANGE / 2
            ball_xyz[:, 1] = -self.TABLE_Y_RANGE / 2
            # Roll towards right
            self.ball_velocity = torch.tensor([0, 1, 0]) * random.uniform(*self.BALL_SPEED_RANGE)
            
        elif edge_choice == 1:  # Right edge
            ball_xyz[:, 0] = (torch.rand(b) * 2 - 1) * self.TABLE_X_RANGE / 2
            ball_xyz[:, 1] = self.TABLE_Y_RANGE / 2
            # Roll towards left
            self.ball_velocity = torch.tensor([0, -1, 0]) * random.uniform(*self.BALL_SPEED_RANGE)
            
        elif edge_choice == 2:  # Front edge (closer to robot)
            ball_xyz[:, 0] = self.TABLE_X_RANGE / 2
            ball_xyz[:, 1] = (torch.rand(b) * 2 - 1) * self.TABLE_Y_RANGE / 2
            # Roll towards back
            self.ball_velocity = torch.tensor([-1, 0, 0]) * random.uniform(*self.BALL_SPEED_RANGE)
            
        else:  # Back edge
            ball_xyz[:, 0] = -self.TABLE_X_RANGE / 2
            ball_xyz[:, 1] = (torch.rand(b) * 2 - 1) * self.TABLE_Y_RANGE / 2
            # Roll towards front
            self.ball_velocity = torch.tensor([1, 0, 0]) * random.uniform(*self.BALL_SPEED_RANGE)
        
        ball_xyz[:, 2] = self.ball_radius
        self.distractor_ball.set_pose(Pose.create_from_pq(ball_xyz))
        
        # Store initial ball position
        self.ball_initial_pos = ball_xyz[0].clone()
        self.ball_start_edge = edge_choice

    def _update_ball_movement(self):
        """Update ball movement if it should be moving"""
        if not self.ball_is_moving and self.current_step >= self.ball_start_step:
            self.ball_is_moving = True
        
        if self.ball_is_moving:
            # Apply velocity to ball
            current_pos = self.distractor_ball.pose.p[0]
            new_pos = current_pos + self.ball_velocity * (1.0 / 60.0)  # Assuming 60 FPS
            
            # Keep ball on table surface
            new_pos[2] = self.ball_radius
            
            # Stop ball if it goes off table
            if (abs(new_pos[0]) > self.TABLE_X_RANGE / 2 + 0.05 or 
                abs(new_pos[1]) > self.TABLE_Y_RANGE / 2 + 0.05):
                self.ball_is_moving = False
            else:
                self.distractor_ball.set_pose(Pose.create_from_pq(new_pos.unsqueeze(0)))

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            target_cube_pose=self.target_cube.pose.raw_pose,
            distractor_ball_pose=self.distractor_ball.pose.raw_pose,
            is_grasping_cube=self.agent.is_grasping(self.target_cube),
            is_grasping_ball=self.agent.is_grasping(self.distractor_ball),
            ball_is_moving=torch.tensor([self.ball_is_moving], dtype=torch.bool, device=self.device),  # Fixed tensor creation
            ball_velocity=self.ball_velocity.to(self.device),  # Ensure proper device
            current_step=torch.tensor([self.current_step], dtype=torch.int32, device=self.device),  # Fixed tensor creation
            ball_start_step=torch.tensor([self.ball_start_step], dtype=torch.int32, device=self.device)  # Fixed tensor creation
        )
        
        return obs

    def evaluate(self):
        """Evaluate if the correct target (red cube) is picked, not the distractor ball"""
        # Check basic task completion for target cube
        is_cube_lifted = self.target_cube.pose.p[:, 2] > self.cube_half_size + self.lift_thresh
        is_cube_grasped = self.agent.is_grasping(self.target_cube)
        is_robot_static = self.agent.is_static(0.2)
        
        # Check if wrong object (ball) was grasped - this should fail the task
        is_ball_grasped = self.agent.is_grasping(self.distractor_ball)
        
        # Success requires grasping cube, not ball
        success = (is_cube_lifted & is_cube_grasped & is_robot_static & (~is_ball_grasped))
        
        return {
            "success": success,
            "is_cube_lifted": is_cube_lifted,
            "is_cube_grasped": is_cube_grasped,
            "is_robot_static": is_robot_static,
            "is_ball_grasped": is_ball_grasped,
            "ball_distracted_robot": is_ball_grasped,  # Indicates if robot was distracted
            "ball_is_moving": torch.tensor(self.ball_is_moving),
            "task_focus_maintained": ~is_ball_grasped  # True if robot stayed focused on cube
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # 1. Reaching reward for target cube
        tcp_to_cube_dist = torch.linalg.norm(
            self.target_cube.pose.p - self.agent.tcp_pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_cube_dist)
        
        # 2. Grasping reward for target cube
        cube_grasping_reward = self.agent.is_grasping(self.target_cube).float() * 2
        
        # 3. Lifting reward for target cube
        lift_height = self.target_cube.pose.p[:, 2] - self.cube_half_size
        lifting_reward = torch.clamp(lift_height / self.lift_thresh, 0, 1) * 2
        
        # 4. Focus maintenance bonus (not being distracted by ball)
        focus_bonus = (~self.agent.is_grasping(self.distractor_ball)).float() * 0.5
        
        # 5. Heavy penalty for grasping wrong object (ball)
        distraction_penalty = self.agent.is_grasping(self.distractor_ball).float() * 4.0
        
        # 6. Ball movement awareness (small bonus for continuing task despite distraction)
        if self.ball_is_moving:
            persistence_bonus = 0.3
        else:
            persistence_bonus = 0.0
        
        reward = (reaching_reward + cube_grasping_reward + lifting_reward + 
                 focus_bonus + persistence_bonus - distraction_penalty)
        
        # Success bonus
        reward[info["success"]] += 4
            
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 10
    
    def step(self, action):
        """Override step to update ball movement"""
        self.current_step += 1
        
        # Update ball movement before physics step
        self._update_ball_movement()
        
        # Call parent step method
        return super().step(action)

DistractorBallPickCubeEnv.__doc__ = DISTRACTOR_BALL_DOC_STRING