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


@register_env("Dynamic-PickCubeWithCollision-Medium", max_episode_steps=200)
class DynamicMediumPickCubeEnv(BaseEnv):
    """
    **Task Description:**
    A pick and lift task with single dynamic interference. The robot must pick up a red cube 
    and lift it above minimum height. During the task, a ball will hit the cube 
    once to test the robot's adaptability to dynamic disturbances.

    **Key Features:**
    - Based on PickCube v1 but with one-time dynamic interference
    - Ball launches once during the episode to hit the target cube
    - Tests robot's ability to adapt to unexpected disturbances

    **Randomizations:**
    - Cube initial position and orientation
    - Ball launch timing (between step 10-50)
    - Ball launch direction and speed

    **Success Conditions:**
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
        interference_step_range=(10, 50),  # When to trigger interference
        ball_speed_range=(1.0, 2.0),       # Ball launch speed range
        **kwargs
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.interference_step_range = interference_step_range
        self.ball_speed_range = ball_speed_range
        
        # Task parameters
        self.cube_half_size = 0.02
        self.lift_thresh = 0.05  # Lift height for success
        self.ball_radius = 0.02  # Slightly larger ball for better collision chance
        
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
        # Build table scene
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # Create target cube (red)
        self.cube_half_size_tensor = common.to_tensor([self.cube_half_size] * 3, device=self.device)
        self.cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[1, 0, 0, 1],  # Red cube
            name="target_cube",
            initial_pose=sapien.Pose(p=[0, 0, 0.1]),
        )

        # Create interference ball (blue) - larger for better collision
        ball_builder = self.scene.create_actor_builder()
        
        # Physics material for realistic bouncing
        ball_material = sapien.pysapien.physx.PhysxMaterial(
            static_friction=0.3,
            dynamic_friction=0.2,
            restitution=0.2
        )
        
        ball_builder.add_sphere_collision(
            radius=self.ball_radius,
            material=ball_material,
            density=500.0  
        )
        ball_builder.add_sphere_visual(
            radius=self.ball_radius,
            material=sapien.render.RenderMaterial(base_color=[0, 0, 1, 1])  # Blue
        )
        ball_builder.initial_pose = sapien.Pose(p=[0, 0, -10])  # Start hidden below table
        self.interference_ball = ball_builder.build(name="interference_ball")

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # Initialize cube position (random on table surface)
            cube_xyz = torch.zeros((b, 3))
            cube_xy = torch.rand((b, 2)) * 0.2 - 0.1  # [-0.1, 0.1] range
            cube_xyz[:, :2] = cube_xy
            cube_xyz[:, 2] = self.cube_half_size  # On table surface
            
            cube_q = randomization.random_quaternions(
                b, lock_x=True, lock_y=True, lock_z=False, device=self.device
            )
            self.cube.set_pose(Pose.create_from_pq(p=cube_xyz, q=cube_q))

            # Hide interference ball initially
            ball_xyz = torch.zeros((b, 3))
            ball_xyz[:, 2] = -10  # Below table, hidden
            self.interference_ball.set_pose(Pose.create_from_pq(p=ball_xyz, q=[1, 0, 0, 0]))

            # Initialize interference tracking - one trigger per episode
            self.interference_state = {
                'step_count': torch.zeros(self.num_envs, dtype=torch.int32, device=self.device),
                'interference_step': torch.randint(
                    self.interference_step_range[0], 
                    self.interference_step_range[1], 
                    (self.num_envs,), 
                    device=self.device, 
                    dtype=torch.int32
                ),
                'interference_triggered': torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
                'ball_active': torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
                'collision_detected': torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            }

    def step(self, action):
        # Execute standard ManiSkill step
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Update step counter
        self.interference_state['step_count'] += 1
        
        # Check if it's time to trigger interference
        self._check_interference_trigger()
        
        # Check for ball-cube collisions
        self._check_ball_cube_collision()
        
        # Update ball state (deactivate if fallen off table)
        self._update_ball_state()
        
        return obs, reward, terminated, truncated, info

    def _check_interference_trigger(self):
        """Check if it's time to launch the interference ball"""
        current_step = self.interference_state['step_count']
        
        # Trigger interference at predetermined step if not already triggered
        should_trigger = (
            (current_step == self.interference_state['interference_step']) &
            (~self.interference_state['interference_triggered'])
        )
        
        if should_trigger.any():
            self._launch_interference_ball(should_trigger)
            self.interference_state['interference_triggered'][should_trigger] = True

    def _check_ball_cube_collision(self):
        """Check for collisions between ball and cube"""
        if not self.interference_state['ball_active'].any():
            return
            
        # Get positions
        ball_pos = self.interference_ball.pose.p
        cube_pos = self.cube.pose.p
        
        # Calculate distance between ball and cube centers
        distance = torch.linalg.norm(ball_pos - cube_pos, axis=1)
        collision_threshold = self.ball_radius + self.cube_half_size + 0.005  # Small buffer
        
        # Detect collision
        collision_mask = (
            (distance < collision_threshold) & 
            self.interference_state['ball_active'] &
            (~self.interference_state['collision_detected'])
        )
        
        if collision_mask.any():
            # Mark collision as detected
            self.interference_state['collision_detected'][collision_mask] = True
            
            # Optional: Add impulse to cube for more dramatic effect
            cube_vel = self.cube.linear_velocity.clone()
            ball_vel = self.interference_ball.linear_velocity
            
            # Transfer some momentum from ball to cube
            momentum_transfer = 0.3
            velocity_addition = ball_vel[collision_mask] * momentum_transfer
            cube_vel[collision_mask] += velocity_addition
            self.cube.set_linear_velocity(cube_vel)

    def _launch_interference_ball(self, launch_mask):
        """Launch interference ball toward cube for specified environments"""
        if not launch_mask.any():
            return
            
        # Get cube positions for environments that need ball launch
        cube_pos = self.cube.pose.p[launch_mask]
        b = launch_mask.sum()
        
        # Calculate launch position closer to cube for better accuracy
        launch_pos = torch.zeros((b, 3), device=self.device)
        
        # Random side selection for each environment
        sides = torch.randint(0, 4, (b,), device=self.device)
        
        # Set launch positions closer to the table and aligned with cube
        for i in range(b):
            cube_x, cube_y = cube_pos[i, 0], cube_pos[i, 1]
            # Reduce noise for better accuracy
            noise_x = torch.randn(1, device=self.device) * 0.02
            noise_y = torch.randn(1, device=self.device) * 0.02
            
            if sides[i] == 0:  # From +X side
                launch_pos[i] = torch.tensor([0.25, cube_y + noise_y, cube_pos[i, 2]], device=self.device)
            elif sides[i] == 1:  # From -X side  
                launch_pos[i] = torch.tensor([-0.25, cube_y + noise_y, cube_pos[i, 2]], device=self.device)
            elif sides[i] == 2:  # From +Y side
                launch_pos[i] = torch.tensor([cube_x + noise_x, 0.25, cube_pos[i, 2]], device=self.device)
            else:  # From -Y side
                launch_pos[i] = torch.tensor([cube_x + noise_x, -0.25, cube_pos[i, 2]], device=self.device)
        
        # Set ball position for launching environments
        current_ball_pos = self.interference_ball.pose.p.clone()
        current_ball_pos[launch_mask] = launch_pos
        self.interference_ball.set_pose(Pose.create_from_pq(p=current_ball_pos, q=[1, 0, 0, 0]))
        
        # Calculate precise launch velocity using physics trajectory
        # Predict cube position accounting for any movement
        cube_vel = self.cube.linear_velocity[launch_mask]
        flight_time = 0.3  # Estimated flight time
        predicted_cube_pos = cube_pos + cube_vel * flight_time
        
        # Calculate trajectory to hit predicted position
        target_pos = predicted_cube_pos.clone()
        target_pos[:, 2] = cube_pos[:, 2]  # Keep at cube height
        
        # Calculate required velocity for ballistic trajectory
        displacement = target_pos - launch_pos
        horizontal_dist = torch.linalg.norm(displacement[:, :2], axis=1)
        vertical_dist = displacement[:, 2]
        
        # Use physics to calculate launch velocity
        gravity = 9.81
        launch_angle = torch.deg2rad(torch.tensor(15.0, device=self.device))  # 15 degree launch angle
        
        # Calculate initial velocity magnitude needed
        v0_squared = (gravity * horizontal_dist**2) / (
            horizontal_dist * torch.tan(launch_angle) - vertical_dist
        )
        v0 = torch.sqrt(torch.clamp(v0_squared, min=1.0))  # Clamp to avoid invalid values
        
        # Calculate velocity components
        v_horizontal = v0 * torch.cos(launch_angle)
        v_vertical = v0 * torch.sin(launch_angle)
        
        # Direction vector (horizontal)
        horizontal_direction = displacement[:, :2] / horizontal_dist.unsqueeze(1)
        
        # Combine into 3D velocity vector
        velocity = torch.zeros((b, 3), device=self.device)
        velocity[:, :2] = horizontal_direction * v_horizontal.unsqueeze(1)
        velocity[:, 2] = v_vertical
        
        # Add small random variation while maintaining accuracy
        velocity_noise = torch.randn_like(velocity) * 0.1
        velocity += velocity_noise
        
        # Apply velocity to ball
        current_vel = self.interference_ball.linear_velocity.clone()
        current_vel[launch_mask] = velocity
        self.interference_ball.set_linear_velocity(current_vel)
        
        # Mark ball as active for these environments
        self.interference_state['ball_active'][launch_mask] = True

    def _update_ball_state(self):
        """Update ball state and deactivate if it falls or moves too far"""
        ball_pos = self.interference_ball.pose.p
        
        # Deactivate ball if it falls below table, moves too far, or after collision
        should_deactivate = (
            (ball_pos[:, 2] < -0.05) |  # Below table level
            (torch.linalg.norm(ball_pos[:, :2], axis=1) > 0.8) |  # Too far from center  
            (self.interference_state['collision_detected'] & 
             (self.interference_state['step_count'] - 
              torch.where(self.interference_state['collision_detected'], 
                         self.interference_state['step_count'], 0) > 10))  # 10 steps after collision
        )
        
        deactivate_mask = should_deactivate & self.interference_state['ball_active']
        
        if deactivate_mask.any():
            # Move deactivated balls below table (hide them)
            hidden_pos = ball_pos.clone()
            hidden_pos[deactivate_mask, 2] = -10
            self.interference_ball.set_pose(Pose.create_from_pq(p=hidden_pos, q=[1, 0, 0, 0]))
            
            # Reset velocity
            zero_vel = torch.zeros_like(self.interference_ball.linear_velocity)
            self.interference_ball.set_linear_velocity(zero_vel)
            
            # Mark as inactive
            self.interference_state['ball_active'][deactivate_mask] = False

    def evaluate(self):
        cube_pos = self.cube.pose.p
        
        # Check if cube is being grasped
        is_cube_grasped = self.agent.is_grasping(self.cube)
        
        # Check if cube is lifted above threshold
        cube_height = cube_pos[:, 2] - self.cube_half_size
        is_cube_lifted = cube_height > self.lift_thresh
        
        # Check if robot is static
        is_robot_static = self.agent.is_static(0.2)
        
        # Check if cube is stable (not moving too fast)
        cube_velocity = torch.linalg.norm(self.cube.linear_velocity, axis=1)
        is_cube_stable = cube_velocity < 0.1
        
        # Success: cube grasped, lifted, robot static, and cube stable
        success = is_cube_grasped & is_cube_lifted & is_robot_static & is_cube_stable
        
        return {
            "is_cube_grasped": is_cube_grasped,
            "is_cube_lifted": is_cube_lifted,
            "is_robot_static": is_robot_static,
            "is_cube_stable": is_cube_stable,
            "cube_height": cube_height,
            "cube_velocity": cube_velocity,
            "interference_triggered": self.interference_state['interference_triggered'],
            "collision_detected": self.interference_state['collision_detected'],
            "ball_active": self.interference_state['ball_active'],
            "success": success.bool(),
        }

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )
        
        if "state" in self.obs_mode:
            obs.update(
                cube_pose=self.cube.pose.raw_pose,
                tcp_to_cube_pos=self.cube.pose.p - self.agent.tcp.pose.p,
                ball_pose=self.interference_ball.pose.raw_pose,
                ball_velocity=self.interference_ball.linear_velocity,
                ball_active=self.interference_state['ball_active'].float(),
                interference_triggered=self.interference_state['interference_triggered'].float(),
                steps_to_interference=(
                    self.interference_state['interference_step'] - self.interference_state['step_count']
                ).float().clamp(min=0),
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # Reward for reaching the cube
        tcp_pos = self.agent.tcp.pose.p
        cube_pos = self.cube.pose.p
        tcp_to_cube_dist = torch.linalg.norm(tcp_pos - cube_pos, axis=1)
        reaching_reward = 1 - torch.tanh(5.0 * tcp_to_cube_dist)
        
        # Grasping reward
        grasping_reward = 3.0 * info["is_cube_grasped"].float()
        
        # Lifting reward
        lifting_reward = 2.0 * torch.clamp(info["cube_height"] / self.lift_thresh, 0, 1)
        
        # Success reward
        success_reward = 10.0 * info["success"].float()
        
        # Adaptation bonus: extra reward for successful manipulation after collision
        adaptation_bonus = 0.0
        if info["collision_detected"].any():
            # Bonus for maintaining control after ball collision
            post_collision_control = info["is_cube_grasped"] & info["collision_detected"]
            adaptation_bonus = 1.0 * post_collision_control.float()
        
        # Stability bonus: reward for achieving stable lift
        stability_bonus = 0.5 * (info["is_cube_stable"] & info["is_cube_lifted"]).float()
        
        total_reward = (
            reaching_reward + grasping_reward + lifting_reward + 
            success_reward + adaptation_bonus + stability_bonus
        )
        
        return total_reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 17.0