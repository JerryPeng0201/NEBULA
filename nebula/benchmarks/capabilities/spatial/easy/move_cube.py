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


@register_env("Spatial-MoveCube-Easy", max_episode_steps=50)
class SpatialEasyMoveCubeEnv(BaseEnv):
    """
    **Task Description:**
    The goal is to pick up a red cube and place it in a specified spatial relationship 
    (left/right/front/back) relative to a green reference cube placed at the center.

    **Randomizations:**
    - red cube has random xy position and z-axis rotation
    - green cube is fixed at center with random z-axis rotation
    - target direction can be specified via options

    **Success Conditions:**
    - the red cube is placed in the correct spatial relationship to the green cube
    - the red cube is static
    - the red cube is not being grasped by the robot (robot must let go of the cube)
    """

    SUPPORTED_ROBOTS = ["panda", "fetch"]
    agent: Union[Panda, Fetch]
    task_instruction = ""

    def __init__(
        self, *args, robot_uids="panda", robot_init_qpos_noise=0, 
        target_direction="right", **kwargs
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.target_direction = target_direction  # "left", "right", "front", "back"
        self.target_offset_distance = 0.08  # 6cm away from green cube
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
        self.cube_half_size = common.to_tensor([0.02] * 3, device=self.device)
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        
        # Red cube (movable)
        self.red_cube = actors.build_cube(
            self.scene,
            half_size=0.02,
            color=[1, 0, 0, 1],
            name="red_cube",
            initial_pose=sapien.Pose(p=[0, 0, 0.1]),
        )
        
        # Green cube (reference, stays at center)
        self.green_cube = actors.build_cube(
            self.scene,
            half_size=0.02,
            color=[0, 1, 0, 1],
            name="green_cube",
            initial_pose=sapien.Pose(p=[0, 0, 0.1]),
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # Update target direction from options if provided
            if "target_direction" in options:
                self.target_direction = options["target_direction"]

            self.task_instruction = f"Pick up the red cube and place it to the near {self.target_direction} of the green cube."

            xyz = torch.zeros((b, 3))
            xyz[:, 2] = 0.02  # on table surface

            # Place green cube at center (reference cube)
            green_xyz = xyz.clone()
            green_xyz[:, 0] = 0.0  # center x
            green_xyz[:, 1] = 0.0  # center y
            self.green_cube.set_pose(Pose.create_from_pq(p=green_xyz))

            # Place red cube at random position (avoid collision with green cube)
            red_xy = torch.rand((b, 2), device=self.device) * 0.8 - 0.4
            # Ensure red cube is not too close to green cube initially
            distance_to_center = torch.linalg.norm(red_xy, axis=1)
            too_close_mask = distance_to_center < 0.15
            while too_close_mask.any():
                red_xy[too_close_mask] = torch.rand((too_close_mask.sum(), 2), device=self.device) * 0.8 - 0.4
                distance_to_center[too_close_mask] = torch.linalg.norm(red_xy[too_close_mask], axis=1)
                too_close_mask = distance_to_center < 0.15

            xyz[:, :2] = red_xy
            self.red_cube.set_pose(Pose.create_from_pq(p=xyz))

    def _get_target_position(self, green_pos):
        """Calculate target position based on direction relative to green cube"""
        target_pos = green_pos.clone()
        
        if self.target_direction == "right":
            target_pos[:, 1] -= self.target_offset_distance 
        elif self.target_direction == "left":
            target_pos[:, 1] += self.target_offset_distance 
        elif self.target_direction == "front":
            target_pos[:, 0] += self.target_offset_distance
        elif self.target_direction == "back":
            target_pos[:, 0] -= self.target_offset_distance
        else:
            raise ValueError(f"Invalid target direction: {self.target_direction}")
        
        return target_pos

    def evaluate(self):
        red_pos = self.red_cube.pose.p
        green_pos = self.green_cube.pose.p
        
        # Calculate target position based on specified direction
        target_pos = self._get_target_position(green_pos)
        
        # Check if red cube is in correct position relative to green cube
        distance_to_target = torch.linalg.norm(red_pos - target_pos, axis=1)
        tolerance = 0.03  # 3cm tolerance
        is_red_in_correct_position = distance_to_target <= tolerance
        
        # Check if red cube is on table (same z-level as green cube)
        z_tolerance = 0.01
        is_red_on_table = torch.abs(red_pos[:, 2] - green_pos[:, 2]) <= z_tolerance
        
        # Other success conditions
        is_red_static = self.red_cube.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        is_red_grasped = self.agent.is_grasping(self.red_cube)
        
        # Overall success
        success = is_red_in_correct_position & is_red_on_table & is_red_static & (~is_red_grasped)
        
        return {
            "is_red_grasped": is_red_grasped,
            "is_red_in_correct_position": is_red_in_correct_position,
            "is_red_on_table": is_red_on_table,
            "is_red_static": is_red_static,
            "distance_to_target": distance_to_target,
            "target_direction": self.target_direction,
            "success": success.bool(),
        }

    def _get_relative_position(self, red_pos, green_pos):
        """Determine the relative position of red cube with respect to green cube"""
        offset = red_pos - green_pos
        
        # Determine dominant direction
        abs_offset = torch.abs(offset)
        
        # Check which axis has larger offset
        x_dominant = abs_offset[:, 0] > abs_offset[:, 1]
        
        # Initialize result tensor
        result = torch.zeros(red_pos.shape[0], dtype=torch.long, device=red_pos.device)
        
        # 0: left, 1: right, 2: front, 3: back
        result[x_dominant & (offset[:, 1] < 0)] = 1  # right
        result[x_dominant & (offset[:, 1] > 0)] = 0  # left
        result[~x_dominant & (offset[:, 0] < 0)] = 3  # back
        result[~x_dominant & (offset[:, 0] > 0)] = 2  # front
        
        return result

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            target_direction_encoded=self._encode_direction(self.target_direction),
        )
        if "state" in self.obs_mode:
            red_pos = self.red_cube.pose.p
            green_pos = self.green_cube.pose.p
            target_pos = self._get_target_position(green_pos)
            relative_pos = self._get_relative_position(red_pos, green_pos)
            
            obs.update(
                red_cube_pose=self.red_cube.pose.raw_pose,
                green_cube_pose=self.green_cube.pose.raw_pose,
                tcp_to_red_pos=red_pos - self.agent.tcp.pose.p,
                tcp_to_green_pos=green_pos - self.agent.tcp.pose.p,
                red_to_green_pos=green_pos - red_pos,
                target_position=target_pos,
                relative_position=relative_pos,
                distance_to_target=info["distance_to_target"],
            )
        return obs

    def _encode_direction(self, direction):
        """Encode direction as integer: left=0, right=1, front=2, back=3"""
        direction_map = {"left": 0, "right": 1, "front": 2, "back": 3}
        return direction_map.get(direction, 0)

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # reaching reward - encourage reaching red cube
        tcp_pose = self.agent.tcp.pose.p
        red_pos = self.red_cube.pose.p
        red_to_tcp_dist = torch.linalg.norm(tcp_pose - red_pos, axis=1)
        reward = 2 * (1 - torch.tanh(5 * red_to_tcp_dist))

        # grasp and place reward
        green_pos = self.green_cube.pose.p
        target_pos = self._get_target_position(green_pos)
        red_to_target_dist = torch.linalg.norm(target_pos - red_pos, axis=1)
        place_reward = 1 - torch.tanh(5.0 * red_to_target_dist)

        reward[info["is_red_grasped"]] = (4 + place_reward)[info["is_red_grasped"]]

        # ungrasp and static reward
        gripper_width = (self.agent.robot.get_qlimits()[0, -1, 1] * 2).to(self.device)
        is_red_grasped = info["is_red_grasped"]
        ungrasp_reward = (
            torch.sum(self.agent.robot.get_qpos()[:, -2:], axis=1) / gripper_width
        )
        ungrasp_reward[~is_red_grasped] = 1.0
        
        v = torch.linalg.norm(self.red_cube.linear_velocity, axis=1)
        av = torch.linalg.norm(self.red_cube.angular_velocity, axis=1)
        static_reward = 1 - torch.tanh(v * 10 + av)
        
        reward[info["is_red_in_correct_position"]] = (
            6 + (ungrasp_reward + static_reward) / 2.0
        )[info["is_red_in_correct_position"]]

        # success reward
        reward[info["success"]] = 8

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 8

    def set_target_direction(self, direction):
        """Set the target direction for the task"""
        if direction not in ["left", "right", "front", "back"]:
            raise ValueError(f"Invalid direction: {direction}. Must be one of: left, right, front, back")
        self.target_direction = direction

    def get_task_instruction(self):
        return self.task_instruction