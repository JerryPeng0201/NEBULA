from typing import Dict, Union

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from nebula.core.embodiment.robots import PandaWristCam
from nebula.core.simulation.engine import BaseEnv
from nebula.core.simulation.utils import randomization
from nebula.core.sensors.camera import CameraConfig
from nebula.utils import common, sapien_utils
from nebula.utils.geometry import rotation_conversions
from nebula.utils.registration import register_env
from nebula.utils.scene_builder.table import TableSceneBuilder
from nebula.utils.structs.pose import Pose
from nebula.utils.structs.types import SimConfig


@register_env("Control-PlugCharger-Hard", max_episode_steps=200)
class ControlPlugChargerHardEnv(BaseEnv):
    """
    **Task Description:**
    The robot must pick up one of the misplaced shapes on the board/kit and insert it into the correct empty slot.

    **Randomizations:**
    - The charger position is randomized on the XY plane on top of the table. The rotation is also randomized
    - The receptacle position is randomized on the XY plane and the rotation is also randomized. Note that the human render camera has its pose
    fixed relative to the receptacle.

    **Success Conditions:**
    - The charger is inserted into the receptacle
    """

    _base_size = [2e-2, 1.5e-2, 1.2e-2]  # charger base half size
    _peg_size = [8e-3, 0.75e-3, 3.2e-3]  # charger peg half size
    _peg_gap = 7e-3  # charger peg gap
    _clearance = 5e-4  # single side clearance
    _receptacle_size = [1e-2, 5e-2, 5e-2]  # receptacle half size

    SUPPORTED_ROBOTS = ["panda"]
    agent: Union[PandaWristCam]
    SUPPORTED_REWARD_MODES = ["none", "sparse"]

    def __init__(
        self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig()

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
        pose = sapien_utils.look_at([0.3, 0.4, 0.1], [0, 0, 0])
        return [
            CameraConfig(
                "render_camera",
                pose=pose,
                width=512,
                height=512,
                fov=1,
                mount=self.receptacle,
            )
        ]

    def _build_charger(self, peg_size, base_size, gap):
        builder = self.scene.create_actor_builder()

        # peg
        mat = sapien.render.RenderMaterial()
        mat.set_base_color([1, 1, 1, 1])
        mat.metallic = 1.0
        mat.roughness = 0.0
        mat.specular = 1.0
        builder.add_box_collision(sapien.Pose([peg_size[0], gap, 0]), peg_size)
        builder.add_box_visual(
            sapien.Pose([peg_size[0], gap, 0]), peg_size, material=mat
        )
        builder.add_box_collision(sapien.Pose([peg_size[0], -gap, 0]), peg_size)
        builder.add_box_visual(
            sapien.Pose([peg_size[0], -gap, 0]), peg_size, material=mat
        )

        # base
        mat = sapien.render.RenderMaterial()
        mat.set_base_color([1, 1, 1, 1])
        mat.metallic = 0.0
        mat.roughness = 0.1
        builder.add_box_collision(sapien.Pose([-base_size[0], 0, 0]), base_size)
        builder.add_box_visual(
            sapien.Pose([-base_size[0], 0, 0]), base_size, material=mat
        )
        builder.initial_pose = sapien.Pose(p=[0, 0, self._base_size[2]])
        return builder.build(name="charger")

    def _build_receptacle(self, peg_size, receptacle_size, gap):
        builder = self.scene.create_actor_builder()

        sy = 0.5 * (receptacle_size[1] - peg_size[1] - gap)
        sz = 0.5 * (receptacle_size[2] - peg_size[2])
        dx = -receptacle_size[0]
        dy = peg_size[1] + gap + sy
        dz = peg_size[2] + sz

        mat = sapien.render.RenderMaterial()
        mat.set_base_color([1, 1, 1, 1])
        mat.metallic = 0.0
        mat.roughness = 0.1

        poses = [
            sapien.Pose([dx, 0, dz]),
            sapien.Pose([dx, 0, -dz]),
            sapien.Pose([dx, dy, 0]),
            sapien.Pose([dx, -dy, 0]),
        ]
        half_sizes = [
            [receptacle_size[0], receptacle_size[1], sz],
            [receptacle_size[0], receptacle_size[1], sz],
            [receptacle_size[0], sy, receptacle_size[2]],
            [receptacle_size[0], sy, receptacle_size[2]],
        ]
        for pose, half_size in zip(poses, half_sizes):
            builder.add_box_collision(pose, half_size)
            builder.add_box_visual(pose, half_size, material=mat)

        # Fill the gap
        pose = sapien.Pose([-receptacle_size[0], 0, 0])
        half_size = [receptacle_size[0], gap - peg_size[1], peg_size[2]]
        builder.add_box_collision(pose, half_size)
        builder.add_box_visual(pose, half_size, material=mat)

        # Add dummy visual for hole
        mat = sapien.render.RenderMaterial()
        mat.set_base_color(sapien_utils.hex2rgba("#DBB539"))
        mat.metallic = 1.0
        mat.roughness = 0.0
        mat.specular = 1.0
        pose = sapien.Pose([-receptacle_size[0], -(gap * 0.5 + peg_size[1]), 0])
        half_size = [receptacle_size[0], peg_size[1], peg_size[2]]
        builder.add_box_visual(pose, half_size, material=mat)
        pose = sapien.Pose([-receptacle_size[0], gap * 0.5 + peg_size[1], 0])
        builder.add_box_visual(pose, half_size, material=mat)
        builder.initial_pose = sapien.Pose(p=[0, 0, 0.1])
        return builder.build_kinematic(name="receptacle")

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.scene_builder = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.scene_builder.build()
        self.charger = self._build_charger(
            self._peg_size,
            self._base_size,
            self._peg_gap,
        )
        self.receptacle = self._build_receptacle(
            [
                self._peg_size[0],
                self._peg_size[1] + self._clearance,
                self._peg_size[2] + self._clearance,
            ],
            self._receptacle_size,
            self._peg_gap,
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.scene_builder.initialize(env_idx)

            # Initialize agent
            if self.agent.uid == "panda_wristcam":
                qpos = torch.tensor(
                    [
                        0.0,
                        np.pi / 8,
                        0,
                        -np.pi * 5 / 8,
                        0,
                        np.pi * 3 / 4,
                        np.pi / 4,
                        0.04,
                        0.04,
                    ]
                )
                qpos = (
                    torch.normal(
                        0,
                        self.robot_init_qpos_noise,
                        (b, len(qpos)),
                        device=self.device,
                    )
                    + qpos
                )
                qpos[:, -2:] = 0.04
                self.agent.robot.set_qpos(qpos)
                self.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))

            # Initialize charger
            xy = randomization.uniform(
                [-0.1, -0.2], [-0.01 - self._peg_size[0] * 2, 0.2], size=(b, 2)
            )
            pos = torch.zeros((b, 3))
            pos[:, :2] = xy
            pos[:, 2] = self._base_size[2]
            ori = randomization.random_quaternions(
                n=b, lock_x=True, lock_y=True, bounds=(-torch.pi / 3, torch.pi / 3)
            )
            self.charger.set_pose(Pose.create_from_pq(pos, ori))

            # Initialize receptacle
            xy = randomization.uniform([0.01, -0.1], [0.1, 0.1], size=(b, 2))
            pos = torch.zeros((b, 3))
            pos[:, :2] = xy
            pos[:, 2] = 0.1
            ori = randomization.random_quaternions(
                n=b,
                lock_x=True,
                lock_y=True,
                bounds=(torch.pi - torch.pi / 8, torch.pi + torch.pi / 8),
            )
            self.receptacle.set_pose(Pose.create_from_pq(pos, ori))

            self.goal_pose = self.receptacle.pose * (
                sapien.Pose(q=euler2quat(0, 0, np.pi))
            )

    @property
    def charger_base_pose(self):
        return self.charger.pose * (sapien.Pose([-self._base_size[0], 0, 0]))

    def _compute_distance(self):
        obj_pose = self.charger.pose
        obj_to_goal_pos = self.goal_pose.p - obj_pose.p
        obj_to_goal_dist = torch.linalg.norm(obj_to_goal_pos, axis=1)

        obj_to_goal_quat = rotation_conversions.quaternion_multiply(
            rotation_conversions.quaternion_invert(self.goal_pose.q), obj_pose.q
        )
        obj_to_goal_axis = rotation_conversions.quaternion_to_axis_angle(
            obj_to_goal_quat
        )
        obj_to_goal_angle = torch.linalg.norm(obj_to_goal_axis, axis=1)
        obj_to_goal_angle = torch.min(
            obj_to_goal_angle, torch.pi * 2 - obj_to_goal_angle
        )

        return obj_to_goal_dist, obj_to_goal_angle

    def evaluate(self):
        obj_to_goal_dist, obj_to_goal_angle = self._compute_distance()
        success = (obj_to_goal_dist <= 5e-3) & (obj_to_goal_angle <= 0.2)
        return dict(
            obj_to_goal_dist=obj_to_goal_dist,
            obj_to_goal_angle=obj_to_goal_angle,
            success=success,
        )

    def _get_obs_extra(self, info: Dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if self.obs_mode_struct.use_state:
            obs.update(
                charger_pose=self.charger.pose.raw_pose,
                receptacle_pose=self.receptacle.pose.raw_pose,
                goal_pose=self.goal_pose.raw_pose,
            )
        return obs

    def compute_dense_reward(self, obs: Dict, action: torch.Tensor, info: Dict):
        """
        Compute dense reward for the plug charger task.
        
        The reward structure encourages:
        1. Approaching the charger
        2. Grasping the charger
        3. Moving the charger towards the goal
        4. Aligning the charger correctly
        5. Inserting the charger into the receptacle
        6. Successful completion
        """
        reward = torch.zeros(self.num_envs, device=self.device)
        
        # Get key positions and poses
        tcp_pose = self.agent.tcp.pose
        charger_pose = self.charger.pose
        goal_pose = self.goal_pose
        receptacle_pose = self.receptacle.pose
        
        # 1. Reaching reward - encourage approaching the charger
        tcp_to_charger_dist = torch.linalg.norm(tcp_pose.p - charger_pose.p, axis=1)
        reaching_reward = 1.0 - torch.tanh(5.0 * tcp_to_charger_dist)
        reward += reaching_reward * 1.0
        
        # 2. Grasping reward - encourage grasping the charger
        is_grasped = self.agent.is_grasping(self.charger)
        grasping_reward = is_grasped.float() * 2.0
        reward += grasping_reward
        
        # Get distance and angle to goal (from evaluate method)
        obj_to_goal_dist, obj_to_goal_angle = self._compute_distance()
        
        # 3. Positioning reward - encourage moving charger towards goal position
        # Only give this reward when the charger is grasped
        positioning_reward = (1.0 - torch.tanh(3.0 * obj_to_goal_dist)) * 2.0
        reward[is_grasped] += positioning_reward[is_grasped]
        
        # 4. Orientation reward - encourage correct alignment
        # Only give this reward when close to goal position (< 2cm)
        close_to_goal = obj_to_goal_dist < 0.02
        orientation_reward = (1.0 - torch.tanh(5.0 * obj_to_goal_angle)) * 1.5
        reward[is_grasped & close_to_goal] += orientation_reward[is_grasped & close_to_goal]
        
        # 5. Insertion reward - encourage fine positioning for insertion
        # Give extra reward when very close and well aligned
        very_close = obj_to_goal_dist < 0.01
        well_aligned = obj_to_goal_angle < 0.1
        insertion_reward = torch.zeros_like(reward)
        insertion_reward[very_close & well_aligned] = 2.0
        reward += insertion_reward
        
        # 6. Height alignment reward - encourage correct Z positioning
        # The charger should be at the right height for insertion
        charger_height = charger_pose.p[:, 2]
        goal_height = goal_pose.p[:, 2]
        height_diff = torch.abs(charger_height - goal_height)
        height_reward = (1.0 - torch.tanh(10.0 * height_diff)) * 0.5
        reward[is_grasped] += height_reward[is_grasped]
        
        # 7. Stability reward - encourage stable grasping
        # Penalize high velocities when grasped and close to goal
        if hasattr(self.charger, 'linear_velocity'):
            charger_velocity = torch.linalg.norm(self.charger.linear_velocity, axis=1)
            stability_reward = torch.exp(-charger_velocity * 5.0) * 0.3
            near_goal = obj_to_goal_dist < 0.03
            reward[is_grasped & near_goal] += stability_reward[is_grasped & near_goal]
        
        # 8. Success bonus - large reward for successful insertion
        success_bonus = info["success"].float() * 10.0
        reward += success_bonus
        
        # 9. Progress-based shaping - reward steady progress
        # Give small bonus for any improvement in position when grasped
        if hasattr(self, '_prev_obj_to_goal_dist'):
            progress = self._prev_obj_to_goal_dist - obj_to_goal_dist
            progress_reward = torch.clamp(progress * 10.0, -0.1, 0.5)
            reward[is_grasped] += progress_reward[is_grasped]
        self._prev_obj_to_goal_dist = obj_to_goal_dist.clone()
        
        # 10. Penalty for dropping the charger
        # If charger falls off table or moves too far from workspace
        charger_z = charger_pose.p[:, 2]
        charger_xy_dist = torch.linalg.norm(charger_pose.p[:, :2], axis=1)
        
        dropped_penalty = torch.zeros_like(reward)
        # Penalize if charger is too low (dropped) or too far away
        dropped_penalty[charger_z < 0.01] = -1.0
        dropped_penalty[charger_xy_dist > 0.5] = -1.0
        reward += dropped_penalty
        
        return reward

    def compute_normalized_dense_reward(self, obs: Dict, action: torch.Tensor, info: Dict):
        """
        Compute normalized dense reward in range [0, 1].
        
        The maximum possible reward is approximately:
        - Reaching: 1.0
        - Grasping: 2.0  
        - Positioning: 2.0
        - Orientation: 1.5
        - Insertion: 2.0
        - Height: 0.5
        - Stability: 0.3
        - Success: 10.0
        - Progress: 0.5
        Total ≈ 20.0
        """
        dense_reward = self.compute_dense_reward(obs, action, info)
        max_reward = 20.0
        return torch.clamp(dense_reward / max_reward, 0.0, 1.0)
