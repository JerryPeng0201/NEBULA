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

@register_env("AdaptationTest-PickAndPlace", max_episode_steps=150)
class PickAndPlaceTestEnv(BaseEnv):
    """
    Test robot's ability to perform sequential tasks with changing instructions.
    - Two cubes on table: red and blue
    - Phase 1: "Pick up the red cube"
    - Phase 2: "Place the red cube on top of the blue cube" (after red is grasped)
    - Success: red cube is stacked on blue cube
    """
    
    SUPPORTED_ROBOTS = ["panda", "fetch"]
    SUPPORTED_REWARD_MODES = ["none", "sparse", "dense"]
    
    agent: Union[Panda, Fetch]
    
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
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, -0.2, 0.2], [0.0, 0.0, 0.2])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
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
        self.lift_height = 0.04  # Lower lift height requirement
        
        # Build table
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        
        # Create single cube (red)
        self.cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[1.0, 0.2, 0.2, 1.0],  # Red
            name="cube",
            initial_pose=sapien.Pose(p=[0, 0, 0.02])
        )
        
        # Task phase tracking
        self.current_phase = "pick"  # "pick" or "release"
        self.current_instruction = "Pick up the cube"
        self.phase_changed = False
        self.phase_change_step = -1
        self.episode_step_count = 0
        self.max_height_achieved = 0.0  # Track maximum height reached
    
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            
            # Reset task phase and counters
            self.current_phase = "pick"
            self.current_instruction = "Pick up the cube"
            self.phase_changed = False
            self.phase_change_step = -1
            self.episode_step_count = 0
            self.max_height_achieved = 0.0
            
            # Random cube position on table
            cube_xyz = torch.zeros((b, 3), device=self.device)
            cube_xyz[:, 0] = torch.rand((b,), device=self.device) * 0.1 + 0.08  # [0.08, 0.18]
            cube_xyz[:, 1] = (torch.rand((b,), device=self.device) - 0.5) * 0.15  # [-0.075, 0.075]
            cube_xyz[:, 2] = self.cube_half_size
            
            # Set pose
            q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(b, 1)
            self.cube.set_pose(Pose.create_from_pq(p=cube_xyz, q=q))
    
    def step(self, action):
        """Override step to change instruction when cube is lifted"""
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Increment step counter
        self.episode_step_count += 1
        
        # Get cube status
        is_grasping = self.agent.is_grasping(self.cube)
        cube_pos = self.cube.pose.p
        cube_height = cube_pos[:, 2].item() if cube_pos.dim() > 1 else cube_pos[2].item()
        
        # Track maximum height
        self.max_height_achieved = max(self.max_height_achieved, cube_height)
        
        # Check if cube is lifted enough (phase transition condition)
        cube_lifted = cube_height > (self.cube_half_size + self.lift_height)
        
        # Transition to place phase when cube is lifted high enough
        if self.current_phase == "pick" and cube_lifted:
            self._change_to_place_phase()
            info["phase_changed"] = True
            info["new_instruction"] = self.current_instruction
            info["new_phase"] = self.current_phase
        
        # Always include current instruction and phase in info
        info["current_instruction"] = self.current_instruction
        info["current_phase"] = self.current_phase
        
        return obs, reward, terminated, truncated, info
    
    def _change_to_place_phase(self):
        """Change from pick phase to release phase"""
        if not self.phase_changed:
            self.current_phase = "release"
            self.current_instruction = "Release the cube"
            self.phase_changed = True
            self.phase_change_step = self.episode_step_count
            # print(f"Phase changed at step {self.episode_step_count}: {self.current_instruction}")
    
    def evaluate(self):
        """Check task success based on current phase"""
        cube_pos = self.cube.pose.p
        
        # Check status
        is_grasping = self.agent.is_grasping(self.cube)
        cube_height = cube_pos[:, 2]
        cube_lifted = cube_height > (self.cube_half_size + self.lift_height)
        
        # Phase-specific success
        if self.current_phase == "pick":
            # Phase 1 success: cube lifted
            phase_success = cube_lifted & is_grasping
        else:  # release phase
            # Phase 2 success: cube released (gripper open)
            phase_success = ~is_grasping
        
        # Overall success: went through both phases and released
        overall_success = self.phase_changed & ~is_grasping
        
        return {
            "success": overall_success,
            "phase_success": phase_success,
            "is_grasping": is_grasping,
            "cube_lifted": cube_lifted,
            "current_phase": self.current_phase,
            "phase_changed": self.phase_changed,
            "max_height": self.max_height_achieved,
        }
    
    def _get_obs_extra(self, info: Dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            current_phase_encoded=self._encode_phase(self.current_phase),
        )
        
        if "state" in self.obs_mode:
            cube_pos = self.cube.pose.p
            
            obs.update(
                cube_pose=self.cube.pose.raw_pose,
                tcp_to_cube_pos=cube_pos - self.agent.tcp.pose.p,
                cube_height=cube_pos[:, 2],
                is_grasping=self.agent.is_grasping(self.cube),
                phase_changed=torch.tensor(self.phase_changed, device=self.device),
            )
        
        return obs
    
    def _encode_phase(self, phase):
        """Encode phase as integer: pick=0, place=1"""
        return 0 if phase == "pick" else 1
    
    def get_task_instruction(self):
        """Return current language instruction for VLA models"""
        return self.current_instruction
    
    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        reward = torch.zeros(self.num_envs, device=self.device)
        
        tcp_pos = self.agent.tcp.pose.p
        cube_pos = self.cube.pose.p
        is_grasping = self.agent.is_grasping(self.cube).float()
        
        if self.current_phase == "pick":
            # Phase 1: Reward for picking up cube
            
            # Reaching reward
            tcp_to_cube_dist = torch.linalg.norm(cube_pos - tcp_pos, dim=1)
            reaching_reward = 1.0 - torch.tanh(5.0 * tcp_to_cube_dist)
            reward += reaching_reward
            
            # Grasping reward
            reward += is_grasping * 3.0
            
            # Lifting reward (reduced requirement)
            lift_progress = (cube_pos[:, 2] - self.cube_half_size) / self.lift_height
            lift_progress = torch.clamp(lift_progress, 0, 1)
            reward += lift_progress * is_grasping * 3.0
            
        else:  # release phase
            # Phase 2: Simply reward for releasing
            
            # Big reward for releasing the cube
            reward += ~is_grasping.bool().float() * 10.0
            
            # Small penalty for still grasping
            reward -= is_grasping * 1.0
        
        # Success bonuses
        if "phase_success" in info:
            reward += info["phase_success"].float() * 5.0
        
        if "success" in info:
            reward += info["success"].float() * 10.0
        
        return reward
    
    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        dense_reward = self.compute_dense_reward(obs, action, info)
        return torch.clamp(dense_reward / 20.0, 0.0, 1.0)