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


@register_env("Dynamic-ShapeSwitchPick-Easy", max_episode_steps=250)
class DynamicShapeSwitchPickEnv(BaseEnv):
    """
    **Task Description:**
    A pick and lift task with dynamic shape switching. The robot must pick up the cube 
    and lift it above minimum height. During the task, the shapes of the two objects will 
    switch (cube becomes sphere, sphere becomes cube) to test the robot's ability to adapt 
    to geometric changes and continue tracking the correct target.

    **Key Features:**
    - Two objects: initially one cube, one sphere
    - Language instruction: "Pick the cube" (implemented elsewhere)
    - Shape switch at step 10-20 to test geometric adaptation
    - Robot must continue tracking the originally cube object after shape change

    **Randomizations:**
    - Initial positions of both objects
    - Shape switch timing
    - Object orientations

    **Success Conditions:**
    - Pick the originally cube object (regardless of current shape)
    - Object is lifted above minimum height (0.05m)
    - Robot is static after completion
    """

    SUPPORTED_ROBOTS = ["panda", "fetch", "panda_wristcam"]
    agent: Union[Panda, Fetch]

    def __init__(
        self,
        *args,
        robot_uids="panda",
        robot_init_qpos_noise=0.02,
        shape_switch_step_range=(10, 20), 
        **kwargs
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.shape_switch_step_range = shape_switch_step_range
        
        # Task parameters
        self.cube_half_size = 0.02
        self.sphere_radius = 0.025  # Slightly larger to have similar volume
        self.lift_thresh = 0.05  # Lift height for success
        
        # Color definitions (keep colors consistent)
        self.target_color = [1, 0, 0, 1]  # Red for target
        
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

        # Create all 4 objects but initially disable some
        self.cube_half_size_tensor = common.to_tensor([self.cube_half_size] * 3, device=self.device)
        
        # Target cube (initially active)
        self.target_cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=self.target_color,
            name="target_cube",
            initial_pose=sapien.Pose(p=[0, 0, 0.1]),
        )
        
        # Target sphere (initially disabled)
        self.target_sphere = self._build_sphere(
            color=self.target_color,
            name="target_sphere",
            initial_pose=sapien.Pose(p=[1000, 1000, 1000])  # Far away
        )
        self.target_sphere.kinematic = True  # Disable physics
        
        # Distractor sphere (initially active)
        self.distractor_sphere = self._build_sphere(
            color=self.target_color,
            name="distractor_sphere",
            initial_pose=sapien.Pose(p=[0.1, 0, 0.1])
        )
        
        # Distractor cube (initially disabled)
        self.distractor_cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=self.target_color,
            name="distractor_cube",
            initial_pose=sapien.Pose(p=[1000, 1000, 1000]),  # Far away
        )
        self.distractor_cube.kinematic = True  # Disable physics
        
        # Set current active object references
        self.target_object = self.target_cube
        self.distractor_object = self.distractor_sphere

    def _build_sphere(self, color, name, initial_pose):
        """Build a sphere actor"""
        builder = self.scene.create_actor_builder()
        
        # Physics material
        material = sapien.pysapien.physx.PhysxMaterial(
            static_friction=0.5,
            dynamic_friction=0.3,
            restitution=0.1
        )
        
        builder.add_sphere_collision(
            radius=self.sphere_radius,
            material=material,
            density=1000.0
        )
        builder.add_sphere_visual(
            radius=self.sphere_radius,
            material=sapien.render.RenderMaterial(base_color=color)
        )
        builder.initial_pose = initial_pose
        return builder.build(name=name)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # Initialize target object position (initially cube)
            target_xyz = torch.zeros((b, 3))
            target_xy = torch.rand((b, 2)) * 0.2 - 0.1  # [-0.1, 0.1] range
            target_xyz[:, :2] = target_xy
            target_xyz[:, 2] = self.cube_half_size  # On table surface
            
            target_q = randomization.random_quaternions(
                b, lock_x=True, lock_y=True, lock_z=False, device=self.device
            )
            
            # Set target object (cube initially)
            self.target_cube.set_pose(Pose.create_from_pq(p=target_xyz, q=target_q))

            # Initialize distractor object position (initially sphere)
            # Ensure it's not too close to target object
            distractor_xyz = torch.zeros((b, 3))
            for i in range(b):
                while True:
                    distractor_xy = torch.rand(2) * 0.2 - 0.1
                    if torch.linalg.norm(distractor_xy - target_xy[i]) > 0.08:  # At least 8cm apart
                        break
                distractor_xyz[i, :2] = distractor_xy
                distractor_xyz[i, 2] = self.sphere_radius  # On table surface (sphere height)
            
            distractor_q = randomization.random_quaternions(
                b, lock_x=True, lock_y=True, lock_z=False, device=self.device
            )
            
            # Set distractor object (sphere initially)
            self.distractor_sphere.set_pose(Pose.create_from_pq(p=distractor_xyz, q=distractor_q))

            # Initialize shape switch tracking
            self.shape_switch_state = {
                'step_count': torch.zeros(self.num_envs, dtype=torch.int32, device=self.device),
                'switch_step': torch.randint(
                    self.shape_switch_step_range[0], 
                    self.shape_switch_step_range[1] + 1,  # +1 to include upper bound
                    (self.num_envs,), 
                    device=self.device, 
                    dtype=torch.int32
                ),
                'shapes_switched': torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
                'target_is_cube': torch.ones(self.num_envs, dtype=torch.bool, device=self.device),  # Initially true
            }
            
            # Update current object references
            self.target_object = self.target_cube
            self.distractor_object = self.distractor_sphere

    def step(self, action):
        # Execute action
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Update step counter
        self.shape_switch_state['step_count'] += 1
        
        # Check if it's time to switch shapes
        self._check_shape_switch()
        
        return obs, reward, terminated, truncated, info

    def _check_shape_switch(self):
        """Check if it's time to switch the shapes of the objects"""
        current_step = self.shape_switch_state['step_count']
        
        # Switch shapes at randomly assigned step if not already switched
        should_switch = (
            (current_step == self.shape_switch_state['switch_step']) &
            (~self.shape_switch_state['shapes_switched'])
        )
        
        if should_switch.any():
            self._switch_object_shapes(should_switch)
            self.shape_switch_state['shapes_switched'][should_switch] = True
            self.shape_switch_state['target_is_cube'][should_switch] = False  # Target is now sphere

    def _switch_object_shapes(self, switch_mask):
        """Switch the shapes by disabling current objects and enabling new ones"""
        if not switch_mask.any():
            return
        
        # Get current states of visible objects
        target_pose = self.target_object.pose
        distractor_pose = self.distractor_object.pose
        target_vel = self.target_object.linear_velocity
        distractor_vel = self.distractor_object.linear_velocity
        
        # Calculate adjusted positions for shape difference
        target_pose_adjusted = target_pose.raw_pose.clone()
        distractor_pose_adjusted = distractor_pose.raw_pose.clone()
        
        # When cube becomes sphere, adjust height (sphere radius vs cube half_size)
        height_adjustment = self.sphere_radius - self.cube_half_size
        target_pose_adjusted[:, 2] += height_adjustment  # Target cube -> sphere (higher)
        distractor_pose_adjusted[:, 2] -= height_adjustment  # Distractor sphere -> cube (lower)
        
        # Disable current objects by making them kinematic and moving far away
        self.target_object.kinematic = True
        self.distractor_object.kinematic = True
        
        # Move to very far location
        far_pose = Pose.create_from_pq(
            p=torch.tensor([[1000, 1000, 1000]], device=self.device).repeat(self.num_envs, 1),
            q=[1, 0, 0, 0]
        )
        self.target_object.set_pose(far_pose)
        self.distractor_object.set_pose(far_pose)
        
        # Enable and position new objects
        self.target_sphere.kinematic = False
        self.target_sphere.set_pose(Pose.create(target_pose_adjusted))
        self.target_sphere.set_linear_velocity(target_vel)
        
        self.distractor_cube.kinematic = False
        self.distractor_cube.set_pose(Pose.create(distractor_pose_adjusted))
        self.distractor_cube.set_linear_velocity(distractor_vel)
        
        # Update current object references
        self.target_object = self.target_sphere
        self.distractor_object = self.distractor_cube

    def evaluate(self):
        # Determine which object should be grasped based on current state (currently cube-shaped object)
        if not self.shape_switch_state['shapes_switched'].any():
            # Before shape switch: target_cube is cube-shaped, should be grasped
            current_cube_object = self.target_cube
            current_sphere_object = self.distractor_sphere
        else:
            # After shape switch: distractor_cube is cube-shaped, should be grasped
            current_cube_object = self.distractor_cube
            current_sphere_object = self.target_sphere

        current_cube_pos = current_cube_object.pose.p

        # Check if the currently cube-shaped object is being grasped
        is_target_grasped = self.agent.is_grasping(current_cube_object)
        
        # Check if wrong object (currently sphere-shaped) is grasped
        is_distractor_grasped = self.agent.is_grasping(current_sphere_object)

        # Check if the cube-shaped object is lifted above threshold
        target_height = current_cube_pos[:, 2] - self.cube_half_size  # Always use cube half_size for cube object
        is_target_lifted = target_height > self.lift_thresh

        # Check if robot is static
        is_robot_static = self.agent.is_static(0.2)

        # Check if cube-shaped object is stable (not moving too fast)
        target_velocity = torch.linalg.norm(current_cube_object.linear_velocity, axis=1)
        is_target_stable = target_velocity < 0.1

        # Check if cube-shaped object is on table (not fallen)
        is_target_on_table = current_cube_pos[:, 2] > -0.1

        # Success: cube-shaped object grasped, lifted, robot static, and stable
        success = is_target_grasped & is_target_lifted & is_robot_static & is_target_stable

        # Additional metrics for analysis
        tcp_pos = self.agent.tcp.pose.p
        tcp_to_target_dist = torch.linalg.norm(tcp_pos - current_cube_pos, axis=1)
        tcp_to_distractor_dist = torch.linalg.norm(tcp_pos - current_sphere_object.pose.p, axis=1)

        # Check if robot is closer to correct target after shape switch
        post_switch_correct_targeting = torch.ones_like(success, dtype=torch.bool)
        switched_envs = self.shape_switch_state['shapes_switched']
        if switched_envs.any():
            # After switch, robot should be closer to the current cube object
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
            "shapes_switched": self.shape_switch_state['shapes_switched'],
            "target_is_cube": self.shape_switch_state['target_is_cube'],
            "post_switch_correct_targeting": post_switch_correct_targeting,
            "success": success.bool(),
        }

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            target_is_cube=self.shape_switch_state['target_is_cube'].float(),
            shapes_switched=self.shape_switch_state['shapes_switched'].float(),
        )
        
        if "state" in self.obs_mode:
            obs.update(
                target_object_pose=self.target_object.pose.raw_pose,
                distractor_object_pose=self.distractor_object.pose.raw_pose,
                tcp_to_target_pos=self.target_object.pose.p - self.agent.tcp.pose.p,
                tcp_to_distractor_pos=self.distractor_object.pose.p - self.agent.tcp.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # Base reward: encourage reaching the TARGET object (originally cube)
        tcp_pos = self.agent.tcp.pose.p
        target_pos = self.target_object.pose.p
        distractor_pos = self.distractor_object.pose.p
        
        tcp_to_target_dist = torch.linalg.norm(tcp_pos - target_pos, axis=1)
        tcp_to_distractor_dist = torch.linalg.norm(tcp_pos - distractor_pos, axis=1)
        
        # Reward for reaching the correct object
        reaching_reward = 1 - torch.tanh(5.0 * tcp_to_target_dist)
        
        # Penalty for reaching wrong object
        distractor_penalty = -0.5 * (1 - torch.tanh(5.0 * tcp_to_distractor_dist))
        
        # Grasping reward (positive for target, negative for distractor)
        target_grasp_reward = 3.0 * info["is_target_grasped"].float()
        distractor_grasp_penalty = -2.0 * info["is_distractor_grasped"].float()
        
        # Lifting reward
        lifting_reward = 2.0 * torch.clamp(info["target_height"] / self.lift_thresh, 0, 1)
        
        # Success reward
        success_reward = 10.0 * info["success"].float()
        
        # Penalty for dropping target object
        drop_penalty = -5.0 * (~info["is_target_on_table"]).float()
        
        # Adaptation bonus: reward for correct targeting after shape switch
        adaptation_bonus = 0.0
        if info["shapes_switched"].any():
            # Bonus for maintaining focus on target object after shape switch
            post_switch_bonus = 2.0 * info["post_switch_correct_targeting"].float()
            adaptation_bonus = post_switch_bonus * info["shapes_switched"].float()
        
        # Shape confusion penalty: extra penalty if robot goes for wrong object after switch
        confusion_penalty = 0.0
        switched_and_wrong = info["shapes_switched"] & info["is_distractor_grasped"]
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