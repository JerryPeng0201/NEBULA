from typing import Any, Dict, Union
import numpy as np
import sapien
import torch
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

@register_env("Dynamic-PressSwitch-Easy", max_episode_steps=200)
class PressLightSwitchEnv(BaseEnv):
    """
    **Task Description:**
    A time-critical manipulation task where the robot must press a switch to change a red light to green within 3 seconds.
    When the episode starts, the light bulb shows red light. The robot has 3 seconds to press the switch to turn it green.
    If the robot fails to press the switch within the time limit, the light turns off (dark) and the episode fails.

    **Randomizations:**
    - Light bulb position is randomized on the table
    - Switch position is randomized within robot's reach
    - Both objects have randomized z-axis rotation

    **Success Conditions:**
    - The light bulb shows green light (switch was pressed in time)
    - The robot is static after pressing the switch
    - Time limit has not expired

    **Failure Conditions:**
    - Time limit (3 seconds) expires before switch is pressed
    - Light bulb turns off (dark/gray)

    **Time Pressure:**
    - Episode starts with red light
    - 3-second countdown begins immediately
    - Must press switch before timer expires
    - Dense reward encourages speed and efficiency
    """

    SUPPORTED_ROBOTS = ["panda", "fetch"]
    agent: Union[Panda, Fetch]

    # Task constants
    TIME_LIMIT = 3.0  # 3 seconds to complete task
    SWITCH_PRESS_THRESHOLD = 0.05  # Distance threshold for switch activation
    SWITCH_SIZE = 0.01  # Half-size of switch cube

    LIGHT_BULB_OFF_PATH = "../../nebula/utils/building/assets/light_bulb/light_bulb_off.glb"
    LIGHT_BULB_RED_PATH = "../../nebula/utils/building/assets/light_bulb/light_bulb_red.glb"
    LIGHT_BULB_GREEN_PATH = "../../nebula/utils/building/assets/light_bulb/light_bulb_green.glb"
    BUTTON_PATH = "../../nebula/utils/building/assets/light_bulb/button.glb"

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        
        # Light states
        self.LIGHT_OFF = 0
        self.LIGHT_RED = 1  
        self.LIGHT_GREEN = 2
        
        # Episode state variables
        self.current_light_state = self.LIGHT_OFF
        self.timer_remaining = self.TIME_LIMIT
        self.switch_pressed = False
        self.episode_success = False
        self.episode_failed = False
        self.step_count = 0
        
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
        pose = sapien_utils.look_at([0.6, 0.6, 0.6], [0.0, 0.0, 0.2])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _build_light_bulb(self, glb_file, name):
        """Build a light bulb actor from GLB file"""
        builder = self.scene.create_actor_builder()

        
        # Add visual from GLB file
        builder.add_visual_from_file(filename=glb_file, scale=[0.01, 0.01, 0.01])

        builder.initial_pose = sapien.Pose(p=[0.15, 0.0, 0.05])
        
        return builder.build_kinematic(name=name)

    def _build_switch(self):
        """Build a switch (button from GLB file)"""
        builder = self.scene.create_actor_builder()
        
        # Keep collision for press detection (adjust size to match your button)
        builder.add_box_collision(half_size=[self.SWITCH_SIZE, self.SWITCH_SIZE, self.SWITCH_SIZE])
        
        # Replace cube visual with button GLB file
        builder.add_visual_from_file(filename=self.BUTTON_PATH, scale=[0.05, 0.05, 0.05])
        print(f"Successfully loaded button GLB: {self.BUTTON_PATH}")
        
        return builder.build_kinematic(name="switch")

    def _load_scene(self, options: dict):
        # Load table scene
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        
        # Create 3 light bulb actors for different states
        # Note: Replace these with your actual GLB file paths
        self.light_off = self._build_light_bulb(self.LIGHT_BULB_OFF_PATH, "light_off")
        self.light_red = self._build_light_bulb(self.LIGHT_BULB_RED_PATH, "light_red")
        self.light_green = self._build_light_bulb(self.LIGHT_BULB_GREEN_PATH, "light_green")
        
        # Create switch
        self.switch = self._build_switch()

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            
            # Reset episode state
            self.current_light_state = self.LIGHT_OFF
            self.timer_remaining = self.TIME_LIMIT
            self.switch_pressed = False
            self.episode_success = False
            self.episode_failed = False
            self.step_count = 0
            
            # Position light bulb on table (back area)
            light_xyz = torch.zeros((b, 3))
            light_xyz[:, 0] = 0.3  # Fixed X position
            light_xyz[:, 1] = 0.0   # Fixed Y position (center)
            light_xyz[:, 2] = 0.05  # Height above table
            
            # Random rotation for light bulb
            light_q = torch.tensor([euler2quat(np.pi/2, 0, 0)], device=self.device).repeat(b, 1)
            light_pose = Pose.create_from_pq(light_xyz, light_q)
            
            # Set poses for all light bulb actors (they'll be in same position)
            self.light_off.set_pose(light_pose)
            self.light_red.set_pose(light_pose)
            self.light_green.set_pose(light_pose)
            
            # Position switch within robot reach (front area)
            switch_xyz = torch.zeros((b, 3))
            switch_xyz[:, 0] = torch.rand((b,)) * 0.1 - 0.1   # [-0.1, 0] - front of table, robot reach
            switch_xyz[:, 1] = (torch.rand((b,)) - 0.5) * 0.15  # [-0.075, 0.075] - near center
            switch_xyz[:, 2] = self.SWITCH_SIZE  # Switch height
            
            # Random rotation for switch
            switch_q = randomization.random_quaternions(b, lock_x=True, lock_y=True, lock_z=False)
            switch_pose = Pose.create_from_pq(switch_xyz, switch_q)
            self.switch.set_pose(switch_pose)
            
            # Set initial light state to RED
            self._set_light_state(self.LIGHT_OFF)

    def _set_light_state(self, state):
        """Change which light bulb is visible"""
        # Hide all lights first
        self.light_off.hide_visual()
        self.light_red.hide_visual()
        self.light_green.hide_visual()
        
        # Show the appropriate light
        if state == self.LIGHT_OFF:
            self.light_off.show_visual()
        elif state == self.LIGHT_RED:
            self.light_red.show_visual()
        elif state == self.LIGHT_GREEN:
            self.light_green.show_visual()
        
        self.current_light_state = state

    def _check_switch_press(self):
        """Check if robot is close enough to touch the switch"""
        tcp_pos = self.agent.tcp.pose.p
        switch_pos = self.switch.pose.p
        distance = torch.linalg.norm(tcp_pos - switch_pos, axis=1)
        
        # Increased threshold for "touching" rather than "pressing"
        return distance <= self.SWITCH_PRESS_THRESHOLD  # Increased from 0.01 to 0.05

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Increment step counter
        self.step_count += 1
        
        # Light state transitions based on step count
        if self.step_count == 10:
            # Change from OFF to RED after 10 steps
            self._set_light_state(self.LIGHT_RED)
            # print("Step 10: Light changed from OFF to RED")
        
        elif self.step_count >= 110:
            # After 100 steps of RED (step 110), turn OFF if button not pressed
            if not self.switch_pressed:
                self._set_light_state(self.LIGHT_OFF)
                self.episode_failed = True
                # print("Step 110: Time expired! Light turned OFF")
        
        # Check for switch press (only when light is RED)
        elif self.current_light_state == self.LIGHT_RED and not self.switch_pressed:
            if self._check_switch_press().any():
                self.switch_pressed = True
                self._set_light_state(self.LIGHT_GREEN)
                self.episode_success = True
                print(f"Step {self.step_count}: Switch pressed! Light turned GREEN")
        
        return obs, reward, terminated, truncated, info

    def evaluate(self):
        """Evaluate success/failure conditions"""
        # Success: switch was pressed (light is green) and robot is static
        is_switch_pressed = torch.tensor([self.switch_pressed], device=self.device)
        is_light_green = torch.tensor([self.current_light_state == self.LIGHT_GREEN], device=self.device)
        is_robot_static = self.agent.is_static(0.2)
        
        # Failure: time expired (light is off)
        is_time_expired = torch.tensor([self.timer_remaining <= 0], device=self.device)
        is_light_off = torch.tensor([self.current_light_state == self.LIGHT_OFF], device=self.device)
        
        success = is_switch_pressed & is_light_green & is_robot_static
        failed = is_time_expired & is_light_off
        
        return {
            "success": success,
            "failed": failed,
            "is_switch_pressed": is_switch_pressed,
            "is_light_green": is_light_green,
            "is_light_off": is_light_off,
            "is_robot_static": is_robot_static,
            "timer_remaining": torch.tensor([self.timer_remaining], device=self.device),
        }

    def _get_obs_extra(self, info: Dict):
        """Additional observations"""
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            switch_pos=self.switch.pose.p,
            light_state=torch.tensor([self.current_light_state], device=self.device),
            timer_remaining=torch.tensor([self.timer_remaining], device=self.device),
            switch_pressed=torch.tensor([self.switch_pressed], device=self.device),
        )
        
        if "state" in self.obs_mode:
            obs.update(
                tcp_to_switch_pos=self.switch.pose.p - self.agent.tcp.pose.p,
                tcp_to_light_pos=self.light_red.pose.p - self.agent.tcp.pose.p,  # Use red light position as reference
            )
        
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        """Compute dense reward for training"""
        reward = torch.zeros(self.num_envs, device=self.device)
        
        # Base reward for approaching switch
        tcp_pos = self.agent.tcp.pose.p
        switch_pos = self.switch.pose.p
        tcp_to_switch_dist = torch.linalg.norm(tcp_pos - switch_pos, axis=1)
        
        # Dense reaching reward (closer to switch = higher reward)
        reaching_reward = 2.0 * (1 - torch.tanh(5.0 * tcp_to_switch_dist))
        reward += reaching_reward
        
        # Time pressure reward (remaining time bonus)
        time_ratio = max(0, self.timer_remaining / self.TIME_LIMIT)
        time_bonus = 1.0 * time_ratio
        reward += time_bonus
        
        # Switch press reward
        if self.switch_pressed:
            reward += 5.0  # Major reward for successful switch press
            
            # Robot static reward (encourage stopping after pressing)
            if self.agent.is_static(0.2).any():
                reward += 2.0
        
        # Success reward
        if info["success"].any():
            reward += 10.0  # Maximum reward for task completion
        
        # Failure penalty
        if info["failed"].any():
            reward -= 3.0  # Penalty for timeout
        
        # Small step penalty to encourage efficiency
        reward -= 0.01
        
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        """Normalized reward (0-1 scale)"""
        max_reward = 20.0  # Approximate maximum possible reward
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward