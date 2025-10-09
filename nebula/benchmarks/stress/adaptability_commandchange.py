from typing import Any, Dict, Union
import numpy as np
import sapien
import torch

from nebula.core.embodiment.robots import Fetch, Panda
from nebula.core.simulation.engine import BaseEnv
from nebula.core.sensors.camera import CameraConfig
from nebula.utils import common, sapien_utils
from nebula.utils.building import actors
from nebula.utils.registration import register_env
from nebula.utils.scene_builder.table import TableSceneBuilder
from nebula.utils.structs.pose import Pose
from nebula.utils.structs.types import Array, GPUMemoryConfig, SimConfig

@register_env("AdaptationTest-InstructionChange", max_episode_steps=100)
class InstructionChangeTestEnv(BaseEnv):
    """
    Test robot's ability to adapt when language instruction changes during execution.
    - Two cubes on table: blue and green
    - Initial instruction: "Pick up the blue cube"
    - After 10 steps: instruction changes to "Pick up the green cube"
    - Success: robot picks up the green cube (not blue)
    """
    
    SUPPORTED_ROBOTS = ["panda", "fetch"]
    SUPPORTED_REWARD_MODES = ["none", "sparse", "dense"]
    
    agent: Union[Panda, Fetch]
    
    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)
    
    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, -0.2, 0.2], [0.0, 0.0, 0.2])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
        )
    
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
    
    def _load_scene(self, options: dict):
        self.cube_half_size = 0.02
        self.lift_height = 0.05
        self.instruction_change_step = 10
        
        # Build table
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        
        # Create blue cube
        self.blue_cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[0.0, 0.0, 1.0, 1.0],  # Blue
            name="blue_cube",
            initial_pose=sapien.Pose(p=[0, 0, 0.02])
        )
        
        # Create green cube
        self.green_cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[0.0, 1.0, 0.0, 1.0],  # Green
            name="green_cube",
            initial_pose=sapien.Pose(p=[0, 0, 0.02])
        )
        
        # Tracking variables
        self.episode_step_count = 0
        self.instruction_changed = False
        self.instruction_change_triggered_at = -1
        self.current_target = "blue"  # Start with blue
        self.current_instruction = "Pick up the blue cube"
    
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            
            # Reset tracking
            self.episode_step_count = 0
            self.instruction_changed = False
            self.instruction_change_triggered_at = -1
            self.current_target = "blue"
            self.current_instruction = "Pick up the blue cube"
            
            # Position blue cube on the left
            blue_xyz = torch.zeros((b, 3), device=self.device)
            blue_xyz[:, 0] = torch.rand((b,), device=self.device) * 0.05 + 0.10  # [0.10, 0.15]
            blue_xyz[:, 1] = torch.rand((b,), device=self.device) * 0.05 - 0.10  # [-0.10, -0.05]
            blue_xyz[:, 2] = self.cube_half_size
            
            # Position green cube on the right
            green_xyz = torch.zeros((b, 3), device=self.device)
            green_xyz[:, 0] = torch.rand((b,), device=self.device) * 0.05 + 0.10  # [0.10, 0.15]
            green_xyz[:, 1] = torch.rand((b,), device=self.device) * 0.05 + 0.05  # [0.05, 0.10]
            green_xyz[:, 2] = self.cube_half_size
            
            # Set poses
            q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(b, 1)
            self.blue_cube.set_pose(Pose.create_from_pq(p=blue_xyz, q=q))
            self.green_cube.set_pose(Pose.create_from_pq(p=green_xyz, q=q))
    
    def step(self, action):
        """Override step to change instruction at step 10"""
        obs, reward, terminated, truncated, info = super().step(action)
        
        self.episode_step_count += 1
        
        # Change instruction at specified step
        if self.episode_step_count == self.instruction_change_step and not self.instruction_changed:
            self._change_instruction()
            self.instruction_changed = True
            self.instruction_change_triggered_at = self.episode_step_count
            info["instruction_changed"] = True
            info["new_instruction"] = self.current_instruction
            info["new_target"] = self.current_target
        
        # Always include current instruction in info
        info["current_instruction"] = self.current_instruction
        info["current_target"] = self.current_target
        
        return obs, reward, terminated, truncated, info
    
    def _change_instruction(self):
        """Change the instruction from blue to green"""
        self.current_target = "green"
        self.current_instruction = "Pick up the green cube"
        # print(f"Instruction changed at step {self.episode_step_count}: {self.current_instruction}")
    
    def evaluate(self):
        """Check task success based on current target"""
        blue_pos = self.blue_cube.pose.p
        green_pos = self.green_cube.pose.p
        
        # Check which cube is grasped
        is_grasping_blue = self.agent.is_grasping(self.blue_cube)
        is_grasping_green = self.agent.is_grasping(self.green_cube)
        
        # Check which cube is lifted
        blue_lifted = blue_pos[:, 2] > (self.cube_half_size + self.lift_height)
        green_lifted = green_pos[:, 2] > (self.cube_half_size + self.lift_height)
        
        # Robot stability
        is_robot_static = self.agent.is_static(0.2)
        
        # Success based on current target
        if self.current_target == "blue":
            target_grasped = is_grasping_blue
            target_lifted = blue_lifted
        else:  # green
            target_grasped = is_grasping_green
            target_lifted = green_lifted
        
        success = target_grasped & target_lifted & is_robot_static
        
        # Check adaptation success (correct target after change)
        adaptation_success = False
        if self.instruction_changed:
            # Success only if green is picked after instruction change
            adaptation_success = is_grasping_green & green_lifted & is_robot_static
        
        return {
            "success": success,
            "is_grasping_blue": is_grasping_blue,
            "is_grasping_green": is_grasping_green,
            "blue_lifted": blue_lifted,
            "green_lifted": green_lifted,
            "is_robot_static": is_robot_static,
            "instruction_changed": self.instruction_changed,
            "current_target": self.current_target,
            "adaptation_success": adaptation_success,
            "wrong_target_picked": is_grasping_blue & self.instruction_changed,
        }
    
    def _get_obs_extra(self, info: Dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            current_instruction_encoded=self._encode_instruction(self.current_target),
        )
        
        if "state" in self.obs_mode:
            obs.update(
                blue_cube_pose=self.blue_cube.pose.raw_pose,
                green_cube_pose=self.green_cube.pose.raw_pose,
                tcp_to_blue_pos=self.blue_cube.pose.p - self.agent.tcp.pose.p,
                tcp_to_green_pos=self.green_cube.pose.p - self.agent.tcp.pose.p,
                is_grasping_blue=self.agent.is_grasping(self.blue_cube),
                is_grasping_green=self.agent.is_grasping(self.green_cube),
                episode_step=torch.tensor(self.episode_step_count, device=self.device),
                instruction_changed=torch.tensor(self.instruction_changed, device=self.device),
            )
        
        return obs
    
    def _encode_instruction(self, target):
        """Encode instruction as integer: blue=0, green=1"""
        return 0 if target == "blue" else 1
    
    def get_task_instruction(self):
        """Return current language instruction for VLA models"""
        return self.current_instruction
    
    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        reward = torch.zeros(self.num_envs, device=self.device)
        
        tcp_pos = self.agent.tcp.pose.p
        blue_pos = self.blue_cube.pose.p
        green_pos = self.green_cube.pose.p
        
        # Get current target cube position
        if self.current_target == "blue":
            target_pos = blue_pos
            wrong_pos = green_pos
            is_grasping_target = self.agent.is_grasping(self.blue_cube).float()
            is_grasping_wrong = self.agent.is_grasping(self.green_cube).float()
        else:
            target_pos = green_pos
            wrong_pos = blue_pos
            is_grasping_target = self.agent.is_grasping(self.green_cube).float()
            is_grasping_wrong = self.agent.is_grasping(self.blue_cube).float()
        
        # 1. Reaching reward for correct target
        tcp_to_target_dist = torch.linalg.norm(target_pos - tcp_pos, dim=1)
        reaching_reward = 1.0 - torch.tanh(5.0 * tcp_to_target_dist)
        reward += reaching_reward
        
        # 2. Grasping reward for correct target
        reward += is_grasping_target * 3.0
        
        # 3. Penalty for grasping wrong target
        reward -= is_grasping_wrong * 5.0
        
        # 4. Lifting reward for correct target
        lift_progress = (target_pos[:, 2] - self.cube_half_size) / self.lift_height
        lift_progress = torch.clamp(lift_progress, 0, 1)
        reward += lift_progress * is_grasping_target * 2.0
        
        # 5. Adaptation bonus after instruction change
        if self.instruction_changed:
            steps_since_change = self.episode_step_count - self.instruction_change_triggered_at
            if steps_since_change < 10:
                # Bonus for quickly adapting to new instruction
                adaptation_speed_bonus = (10 - steps_since_change) / 10.0
                reward += adaptation_speed_bonus * is_grasping_target
        
        # 6. Success bonus
        if "success" in info:
            reward += info["success"].float() * 10.0
        
        if "adaptation_success" in info:
            reward += info["adaptation_success"].float() * 5.0
        
        return reward
    
    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        dense_reward = self.compute_dense_reward(obs, action, info)
        return torch.clamp(dense_reward / 10.0, 0.0, 1.0)