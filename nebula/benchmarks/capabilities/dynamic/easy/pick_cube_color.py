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

@register_env("Dynamic-ColorSwitchPickCube-Easy", max_episode_steps=250)
class ColorSwitchPickCubeEnv(BaseEnv):
    """
    **Task Description:**
    A pick and lift task with dynamic color switching. The robot must pick up the red cube
    and lift it above minimum height. During the task, the colors of the two cubes will
    switch (red becomes green, green becomes red) to test the robot's ability to adapt
    to visual changes and continue tracking the correct target.

    **Key Features:**
    - Two cubes: initially one red, one green
    - Language instruction: "Pick the red cube" (implemented elsewhere)
    - Color switch at step 10-20 to test visual adaptation
    - Robot must continue tracking the originally red cube after color change

    **Randomizations:**
    - Initial positions of both cubes
    - Color switch timing (between step 10-20)
    - Cube orientations

    **Success Conditions:**
    - Pick the originally red cube (regardless of current color)
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
        color_switch_step_range=(10, 20),
        **kwargs
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.color_switch_step_range = color_switch_step_range

        # Task parameters
        self.cube_half_size = 0.02
        self.lift_thresh = 0.05  # Lift height for success

        # Color definitions
        self.red_color = [1, 0, 0, 1]
        self.green_color = [0, 1, 0, 1]

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
        pose = sapien_utils.look_at([0.6, -0.2, 0.2], [0.0, 0.0, 0.2])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
        )

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        # Build table scene
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # Create 4 cubes for color switching: red target, green target, red distractor, green distractor
        self.cube_half_size_tensor = common.to_tensor([self.cube_half_size] * 3, device=self.device)

        # Red target cube (initially visible)
        self.target_cube_red = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=self.red_color,
            name="target_cube_red",
            initial_pose=sapien.Pose(p=[0, 0, 0.1]),
        )

        # Green target cube (initially hidden)
        self.target_cube_green = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=self.green_color,
            name="target_cube_green",
            initial_pose=sapien.Pose(p=[0, 0, -10]),  # Hidden below table
        )

        # Green distractor cube (initially visible)
        self.distractor_cube_green = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=self.green_color,
            name="distractor_cube_green",
            initial_pose=sapien.Pose(p=[0.1, 0, 0.1]),
        )

        # Red distractor cube (initially hidden)
        self.distractor_cube_red = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=self.red_color,
            name="distractor_cube_red",
            initial_pose=sapien.Pose(p=[0.1, 0, -10]),  # Hidden below table
        )

        # Set current active cube references
        self.target_cube = self.target_cube_red
        self.distractor_cube = self.distractor_cube_green

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # Initialize target cube position (initially red)
            target_xyz = torch.zeros((b, 3))
            target_xy = torch.rand((b, 2)) * 0.2 - 0.1  # [-0.1, 0.1] range
            target_xyz[:, :2] = target_xy
            target_xyz[:, 2] = self.cube_half_size  # On table surface

            target_q = randomization.random_quaternions(
                b, lock_x=True, lock_y=True, lock_z=False, device=self.device
            )

            # Set visible target cube (red initially)
            self.target_cube_red.set_pose(Pose.create_from_pq(p=target_xyz, q=target_q))
            # Hide green target cube
            hidden_pos = torch.zeros((b, 3))
            hidden_pos[:, 2] = -10
            self.target_cube_green.set_pose(Pose.create_from_pq(p=hidden_pos, q=target_q))

            # Initialize distractor cube position (initially green)
            # Ensure it's not too close to target cube
            distractor_xyz = torch.zeros((b, 3))
            for i in range(b):
                while True:
                    distractor_xy = torch.rand(2) * 0.2 - 0.1
                    if torch.linalg.norm(distractor_xy - target_xy[i]) > 0.08:  # At least 8cm apart
                        break
                distractor_xyz[i, :2] = distractor_xy
                distractor_xyz[i, 2] = self.cube_half_size

            distractor_q = randomization.random_quaternions(
                b, lock_x=True, lock_y=True, lock_z=False, device=self.device
            )

            # Set visible distractor cube (green initially)
            self.distractor_cube_green.set_pose(Pose.create_from_pq(p=distractor_xyz, q=distractor_q))
            # Hide red distractor cube
            self.distractor_cube_red.set_pose(Pose.create_from_pq(p=hidden_pos, q=distractor_q))

            # Initialize color switch tracking
            self.color_switch_state = {
                'step_count': torch.zeros(self.num_envs, dtype=torch.int32, device=self.device),
                'switch_step': torch.randint(
                    self.color_switch_step_range[0],
                    self.color_switch_step_range[1] + 1,  # +1 to include upper bound
                    (self.num_envs,),
                    device=self.device,
                    dtype=torch.int32
                ),
                'colors_switched': torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
                'target_is_red': torch.ones(self.num_envs, dtype=torch.bool, device=self.device),  # Initially true
            }

            # Update current cube references
            self.target_cube = self.target_cube_red
            self.distractor_cube = self.distractor_cube_green

    def step(self, action):
        # Execute standard ManiSkill step
        obs, reward, terminated, truncated, info = super().step(action)

        # Update step counter
        self.color_switch_state['step_count'] += 1

        # Check if it's time to switch colors
        self._check_color_switch()

        return obs, reward, terminated, truncated, info

    def _check_color_switch(self):
        """Check if it's time to switch the colors of the cubes"""
        current_step = self.color_switch_state['step_count']

        # Switch colors at randomly assigned step if not already switched
        should_switch = (
            (current_step == self.color_switch_state['switch_step']) &
            (~self.color_switch_state['colors_switched'])
        )

        if should_switch.any():
            self._switch_cube_colors(should_switch)
            self.color_switch_state['colors_switched'][should_switch] = True
            self.color_switch_state['target_is_red'][should_switch] = False  # Target is now green

    def _switch_cube_colors(self, switch_mask):
        """Switch the colors of both cubes by swapping visible/hidden cubes"""
        if not switch_mask.any():
            return

        # Get current positions and states of visible cubes
        target_pose = self.target_cube.pose
        distractor_pose = self.distractor_cube.pose
        target_vel = self.target_cube.linear_velocity
        distractor_vel = self.distractor_cube.linear_velocity
        target_ang_vel = self.target_cube.angular_velocity
        distractor_ang_vel = self.distractor_cube.angular_velocity

        # Hide current cubes (move to below table)
        hidden_pose = Pose.create_from_pq(
            p=torch.tensor([[0, 0, -10]], device=self.device).repeat(self.num_envs, 1),
            q=[1, 0, 0, 0]
        )
        self.target_cube.set_pose(hidden_pose)
        self.distractor_cube.set_pose(hidden_pose)
        
        # Stop the hidden cubes
        zero_vel = torch.zeros_like(target_vel)
        self.target_cube.set_linear_velocity(zero_vel)
        self.target_cube.set_angular_velocity(zero_vel)
        self.distractor_cube.set_linear_velocity(zero_vel)
        self.distractor_cube.set_angular_velocity(zero_vel)

        # Show color-swapped cubes at the EXACT original positions
        self.target_cube_green.set_pose(target_pose)
        self.distractor_cube_red.set_pose(distractor_pose)

        # Set velocities to maintain motion continuity
        self.target_cube_green.set_linear_velocity(target_vel)
        self.target_cube_green.set_angular_velocity(target_ang_vel)
        self.distractor_cube_red.set_linear_velocity(distractor_vel)
        self.distractor_cube_red.set_angular_velocity(distractor_ang_vel)

        # Update current cube references
        self.target_cube = self.target_cube_green
        self.distractor_cube = self.distractor_cube_red

    def evaluate(self):
        # Determine which cube should be grasped (the currently red cube) based on current state
        if not self.color_switch_state['colors_switched'].any():
            # Before color switch: target_cube_red is red, should be grasped
            current_red_cube = self.target_cube_red
            current_green_cube = self.distractor_cube_green
        else:
            # After color switch: distractor_cube_red is red, should be grasped
            current_red_cube = self.distractor_cube_red  
            current_green_cube = self.target_cube_green

        current_red_pos = current_red_cube.pose.p

        # Check if the currently red cube is being grasped
        is_target_grasped = self.agent.is_grasping(current_red_cube)
        
        # Check if wrong cube (currently green) is grasped
        is_distractor_grasped = self.agent.is_grasping(current_green_cube)

        # Check if the red cube is lifted above threshold
        target_height = current_red_pos[:, 2] - self.cube_half_size
        is_target_lifted = target_height > self.lift_thresh

        # Check if robot is static
        is_robot_static = self.agent.is_static(0.2)

        # Check if red cube is stable (not moving too fast)
        target_velocity = torch.linalg.norm(current_red_cube.linear_velocity, axis=1)
        is_target_stable = target_velocity < 0.1

        # Check if red cube is on table (not fallen)
        is_target_on_table = current_red_pos[:, 2] > -0.1

        # Success: red cube grasped, lifted, robot static, and stable
        success = is_target_grasped & is_target_lifted & is_robot_static & is_target_stable

        # Additional metrics for analysis
        tcp_pos = self.agent.tcp.pose.p
        tcp_to_target_dist = torch.linalg.norm(tcp_pos - current_red_pos, axis=1)
        tcp_to_distractor_dist = torch.linalg.norm(tcp_pos - current_green_cube.pose.p, axis=1)

        # Check if robot is closer to correct target after color switch
        post_switch_correct_targeting = torch.ones_like(success, dtype=torch.bool)
        switched_envs = self.color_switch_state['colors_switched']
        if switched_envs.any():
            # After switch, robot should be closer to the current red cube
            post_switch_correct_targeting[switched_envs] = (
                tcp_to_target_dist[switched_envs] < tcp_to_distractor_dist[switched_envs]
            )

        return {
            "is_target_grasped": is_target_grasped,
            "is_distractor_grasped": is_distractor_grasped,
            "is_target_lifted": is_target_lifted,
            "is_target_on_table": is_target_on_table,
            "is_target_stable": is_target_stable,
            "is_robot_static": is_robot_static,
            "target_height": target_height,
            "target_velocity": target_velocity,
            "tcp_to_target_dist": tcp_to_target_dist,
            "tcp_to_distractor_dist": tcp_to_distractor_dist,
            "colors_switched": self.color_switch_state['colors_switched'],
            "target_is_red": self.color_switch_state['target_is_red'],
            "post_switch_correct_targeting": post_switch_correct_targeting,
            "success": success.bool(),
        }

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            target_is_red=self.color_switch_state['target_is_red'].float(),
            colors_switched=self.color_switch_state['colors_switched'].float(),
        )

        if "state" in self.obs_mode:
            obs.update(
                target_cube_pose=self.target_cube.pose.raw_pose,
                distractor_cube_pose=self.distractor_cube.pose.raw_pose,
                tcp_to_target_pos=self.target_cube.pose.p - self.agent.tcp.pose.p,
                tcp_to_distractor_pos=self.distractor_cube.pose.p - self.agent.tcp.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # Base reward: encourage reaching the TARGET cube (originally red)
        tcp_pos = self.agent.tcp.pose.p
        target_pos = self.target_cube.pose.p
        distractor_pos = self.distractor_cube.pose.p

        tcp_to_target_dist = torch.linalg.norm(tcp_pos - target_pos, axis=1)
        tcp_to_distractor_dist = torch.linalg.norm(tcp_pos - distractor_pos, axis=1)

        # Reward for reaching the correct cube
        reaching_reward = 1 - torch.tanh(5.0 * tcp_to_target_dist)

        # Penalty for reaching wrong cube
        distractor_penalty = -0.5 * (1 - torch.tanh(5.0 * tcp_to_distractor_dist))

        # Grasping reward (positive for target, negative for distractor)
        target_grasp_reward = 3.0 * info["is_target_grasped"].float()
        distractor_grasp_penalty = -2.0 * info["is_distractor_grasped"].float()

        # Lifting reward
        lifting_reward = 2.0 * torch.clamp(info["target_height"] / self.lift_thresh, 0, 1)

        # Success reward
        success_reward = 10.0 * info["success"].float()

        # Penalty for dropping target cube
        drop_penalty = -5.0 * (~info["is_target_on_table"]).float()

        # Adaptation bonus: reward for correct targeting after color switch
        adaptation_bonus = 0.0
        if info["colors_switched"].any():
            # Bonus for maintaining focus on target cube after color switch
            post_switch_bonus = 2.0 * info["post_switch_correct_targeting"].float()
            adaptation_bonus = post_switch_bonus * info["colors_switched"].float()

        # Visual confusion penalty: extra penalty if robot goes for wrong cube after switch
        confusion_penalty = 0.0
        switched_and_wrong = info["colors_switched"] & info["is_distractor_grasped"]
        confusion_penalty = -3.0 * switched_and_wrong.float()

        # Stability bonus: reward for achieving stable lift
        stability_bonus = 0.5 * (info["is_target_stable"] & info["is_target_lifted"]).float()

        total_reward = (
            reaching_reward + distractor_penalty + target_grasp_reward +
            distractor_grasp_penalty + lifting_reward + success_reward +
            drop_penalty + adaptation_bonus + confusion_penalty + stability_bonus
        )

        return total_reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 18.0