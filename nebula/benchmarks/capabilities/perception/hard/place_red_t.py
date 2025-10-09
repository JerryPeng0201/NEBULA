from typing import Any, Dict, Union

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import sapien
import torch
import torch.random
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


@register_env("Perception-PlaceRedT-Hard", max_episode_steps=50)
class PerceptionPlaceRedTHardEnv(BaseEnv):
    """
    **Task Description:**
    Place any object (blue sphere, green cube, red T-shape, L-shape, yellow cube, or red sphere) into the shallow bin.

    **Randomizations:**
    - The position of the bin and the objects are randomized: The bin is initialized in [0, 0.1] x [-0.1, 0.1],
    and the objects are initialized at fixed positions in [-0.1, -0.05] x [-0.1, 0.1]

    **Success Conditions:**
    - Any object is placed on the top of the bin. The robot remains static and the gripper is not closed at the end state.
    """

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/PlaceSphere-v1_rt.mp4"
    SUPPORTED_ROBOTS = ["panda", "fetch"]

    # Specify some supported robot types
    agent: Union[Panda, Fetch]

    # set common size for objects
    object_size = 0.02  # radius for sphere, half_size for cube
    
    # T-shape properties
    T_mass = 0.1
    T_dynamic_friction = 0.3
    T_static_friction = 0.3
    TARGET_RED = np.array([194, 19, 22, 255]) / 255
    
    # L-shape properties (same scale as T)
    L_mass = 0.1
    L_dynamic_friction = 0.3
    L_static_friction = 0.3
    
    # bin dimensions
    inner_side_half_len = 0.04  # adjusted for objects
    short_side_half_size = 0.0025
    block_half_size = [
        short_side_half_size,
        2 * short_side_half_size + inner_side_half_len,
        2 * short_side_half_size + inner_side_half_len,
    ]
    edge_block_half_size = [
        short_side_half_size,
        2 * short_side_half_size + inner_side_half_len,
        2 * short_side_half_size,
    ]

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

    def _build_bin(self):
        builder = self.scene.create_actor_builder()

        # init the locations of the basic blocks
        dx = self.block_half_size[1] - self.block_half_size[0]
        dy = self.block_half_size[1] - self.block_half_size[0]
        dz = self.edge_block_half_size[2] + self.block_half_size[0]

        # build the bin bottom and edge blocks
        poses = [
            sapien.Pose([0, 0, 0]),
            sapien.Pose([-dx, 0, dz]),
            sapien.Pose([dx, 0, dz]),
            sapien.Pose([0, -dy, dz]),
            sapien.Pose([0, dy, dz]),
        ]
        half_sizes = [
            [self.block_half_size[1], self.block_half_size[2], self.block_half_size[0]],
            self.edge_block_half_size,
            self.edge_block_half_size,
            [
                self.edge_block_half_size[1],
                self.edge_block_half_size[0],
                self.edge_block_half_size[2],
            ],
            [
                self.edge_block_half_size[1],
                self.edge_block_half_size[0],
                self.edge_block_half_size[2],
            ],
        ]
        for pose, half_size in zip(poses, half_sizes):
            builder.add_box_collision(pose, half_size)
            builder.add_box_visual(pose, half_size)

        # build the kinematic bin
        return builder.build_kinematic(name="bin")

    def create_small_tee(self, name="small_tee", target=False, base_color=None):
        """Returns smaller 3D CAD of create_tee - scaled down to match other objects"""
        if base_color is None:
            base_color = self.TARGET_RED
        
        # scaled down dimensions (about 1/3 of original size)
        scale_factor = 0.35
        box1_half_w = (0.2 / 2) * scale_factor  # 0.035
        box1_half_h = (0.05 / 2) * scale_factor  # 0.00875
        half_thickness = (0.04 / 2) * scale_factor if not target else 1e-4  # 0.007

        # scaled center of mass calculation
        com_y = 0.0375 * scale_factor

        builder = self.scene.create_actor_builder()
        first_block_pose = sapien.Pose([0.0, 0.0 - com_y, 0.0])
        first_block_size = [box1_half_w, box1_half_h, half_thickness]
        
        if not target:
            builder._mass = self.T_mass
            tee_material = sapien.pysapien.physx.PhysxMaterial(
                static_friction=self.T_static_friction,
                dynamic_friction=self.T_dynamic_friction,
                restitution=0,
            )
            builder.add_box_collision(
                pose=first_block_pose,
                half_size=first_block_size,
                material=tee_material,
            )
        builder.add_box_visual(
            pose=first_block_pose,
            half_size=first_block_size,
            material=sapien.render.RenderMaterial(
                base_color=base_color,
            ),
        )

        # second block (vertical part)
        second_block_pose = sapien.Pose([0.0, 4 * (box1_half_h) - com_y, 0.0])
        second_block_size = [box1_half_h, (3 / 4) * (box1_half_w), half_thickness]
        if not target:
            builder.add_box_collision(
                pose=second_block_pose,
                half_size=second_block_size,
                material=tee_material,
            )
        builder.add_box_visual(
            pose=second_block_pose,
            half_size=second_block_size,
            material=sapien.render.RenderMaterial(
                base_color=base_color,
            ),
        )
        builder.initial_pose = sapien.Pose(p=[0, 0, 0.1])
        if not target:
            return builder.build_dynamic(name=name)
        else:
            return builder.build_kinematic(name=name)

    def _build_l_shaped_tool(self, name="l_shape_tool"):
        """Build L-shaped tool with same scale as T-shape"""
        builder = self.scene.create_actor_builder()

        # Use same scale factor as T-shape for consistency
        scale_factor = 0.35
        
        # Scale dimensions to match T-shape size
        handle_length = 0.07 * scale_factor  # 0.0245
        hook_length = 0.035 * scale_factor   # 0.01225
        width = 0.02 * scale_factor          # 0.007
        height = 0.04 * scale_factor         # 0.014

        mat = sapien.render.RenderMaterial()
        mat.set_base_color([1, 0, 0, 1])  # Red color
        mat.metallic = 1.0
        mat.roughness = 0.0
        mat.specular = 1.0

        l_material = sapien.pysapien.physx.PhysxMaterial(
            static_friction=self.L_static_friction,
            dynamic_friction=self.L_dynamic_friction,
            restitution=0,
        )

        # Handle part (horizontal)
        builder.add_box_collision(
            sapien.Pose([handle_length / 2, 0, 0]),
            [handle_length / 2, width / 2, height / 2],
            material=l_material,
        )
        builder.add_box_visual(
            sapien.Pose([handle_length / 2, 0, 0]),
            [handle_length / 2, width / 2, height / 2],
            material=mat,
        )

        # Hook part (vertical)
        builder.add_box_collision(
            sapien.Pose([handle_length - hook_length / 2, width, 0]),
            [hook_length / 2, width / 2, height / 2],
            material=l_material,
        )
        builder.add_box_visual(
            sapien.Pose([handle_length - hook_length / 2, width, 0]),
            [hook_length / 2, width / 2, height / 2],
            material=mat,
        )

        builder._mass = self.L_mass
        return builder.build_dynamic(name=name)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        # load the table
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # load the six objects
        self.blue_sphere = actors.build_sphere(
            self.scene,
            radius=self.object_size,
            color=np.array([12, 42, 160, 255]) / 255,  # blue
            name="blue_sphere",
            body_type="dynamic",
            initial_pose=Pose.create_from_pq(p=[0, 0, self.object_size], q=[1, 0, 0, 0])
        )
        
        self.green_cube = actors.build_cube(
            self.scene,
            half_size=self.object_size,
            color=np.array([0, 180, 0, 255]) / 255,  # green
            name="green_cube",
            body_type="dynamic",
        )

        # load the small red T-shape
        self.red_tee = self.create_small_tee(name="red_tee", target=False)

        # load the L-shape
        self.l_shape = self._build_l_shaped_tool(name="l_shape")

        # load additional cube and sphere
        self.yellow_cube = actors.build_cube(
            self.scene,
            half_size=self.object_size,
            color=np.array([255, 255, 0, 255]) / 255,  # yellow
            name="yellow_cube",
            body_type="dynamic",
        )
        
        self.red_sphere = actors.build_sphere(
            self.scene,
            radius=self.object_size,
            color=np.array([255, 0, 0, 255]) / 255,  # red
            name="red_sphere",
            body_type="dynamic",
            initial_pose=Pose.create_from_pq(p=[0, 0, self.object_size], q=[1, 0, 0, 0])
        )

        # load the bin
        self.bin = self._build_bin()

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            # init the table scene
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            xyz = torch.zeros((b, 3))
            q = [1, 0, 0, 0]
            
            # Object positions (6 objects in a single line with proper spacing)
            # All objects on same x-coordinate, spaced in y-direction
            
            # Blue sphere
            xyz[..., 0] = -0.08  # fixed x position for all objects
            xyz[..., 1] = -0.15  # leftmost position
            xyz[..., 2] = self.object_size
            blue_sphere_pose = Pose.create_from_pq(p=xyz.clone(), q=q)
            self.blue_sphere.set_pose(blue_sphere_pose)

            # Green cube
            xyz[..., 0] = -0.08
            xyz[..., 1] = -0.09  # spaced 0.06 units apart
            xyz[..., 2] = self.object_size
            green_cube_pose = Pose.create_from_pq(p=xyz.clone(), q=q)
            self.green_cube.set_pose(green_cube_pose)

            # Red T-shape
            xyz[..., 0] = -0.08
            xyz[..., 1] = -0.03  # spaced 0.06 units apart
            xyz[..., 2] = 0.007  # T thickness/2
            red_tee_pose = Pose.create_from_pq(p=xyz.clone(), q=q)
            self.red_tee.set_pose(red_tee_pose)

            # L-shape
            xyz[..., 0] = -0.08
            xyz[..., 1] = 0.03   # spaced 0.06 units apart
            xyz[..., 2] = 0.007  # L thickness/2
            l_shape_pose = Pose.create_from_pq(p=xyz.clone(), q=q)
            self.l_shape.set_pose(l_shape_pose)

            # Yellow cube
            xyz[..., 0] = -0.08
            xyz[..., 1] = 0.09   # spaced 0.06 units apart
            xyz[..., 2] = self.object_size
            yellow_cube_pose = Pose.create_from_pq(p=xyz.clone(), q=q)
            self.yellow_cube.set_pose(yellow_cube_pose)

            # Red sphere
            xyz[..., 0] = -0.08
            xyz[..., 1] = 0.15   # rightmost position
            xyz[..., 2] = self.object_size
            red_sphere_pose = Pose.create_from_pq(p=xyz, q=q)
            self.red_sphere.set_pose(red_sphere_pose)

            # init the bin in the target zone
            pos = torch.zeros((b, 3))
            pos[:, 0] = (
                torch.rand((b, 1))[..., 0] * 0.1
            )  # x range [0, 0.1]
            pos[:, 1] = (
                torch.rand((b, 1))[..., 0] * 0.2 - 0.1
            )  # y range [-0.1, 0.1]
            pos[:, 2] = self.block_half_size[0]  # on the table
            q = [1, 0, 0, 0]
            bin_pose = Pose.create_from_pq(p=pos, q=q)
            self.bin.set_pose(bin_pose)

    def evaluate(self):
        pos_blue_sphere = self.blue_sphere.pose.p
        pos_green_cube = self.green_cube.pose.p
        pos_red_tee = self.red_tee.pose.p
        pos_l_shape = self.l_shape.pose.p
        pos_yellow_cube = self.yellow_cube.pose.p
        pos_red_sphere = self.red_sphere.pose.p
        pos_bin = self.bin.pose.p
        
        # Check if any object is on the bin
        is_blue_sphere_on_bin = self._check_object_on_bin(pos_blue_sphere, pos_bin, self.object_size)
        is_green_cube_on_bin = self._check_object_on_bin(pos_green_cube, pos_bin, self.object_size)
        is_red_tee_on_bin = self._check_object_on_bin(pos_red_tee, pos_bin, 0.007)
        is_l_shape_on_bin = self._check_object_on_bin(pos_l_shape, pos_bin, 0.007)
        is_yellow_cube_on_bin = self._check_object_on_bin(pos_yellow_cube, pos_bin, self.object_size)
        is_red_sphere_on_bin = self._check_object_on_bin(pos_red_sphere, pos_bin, self.object_size)
        
        is_any_object_on_bin = torch.logical_or(
            torch.logical_or(
                torch.logical_or(is_blue_sphere_on_bin, is_green_cube_on_bin),
                torch.logical_or(is_red_tee_on_bin, is_l_shape_on_bin)
            ),
            torch.logical_or(is_yellow_cube_on_bin, is_red_sphere_on_bin)
        )
        
        # Check if objects are static
        is_blue_sphere_static = self.blue_sphere.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        is_green_cube_static = self.green_cube.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        is_red_tee_static = self.red_tee.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        is_l_shape_static = self.l_shape.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        is_yellow_cube_static = self.yellow_cube.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        is_red_sphere_static = self.red_sphere.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        
        # Check if objects are grasped
        is_blue_sphere_grasped = self.agent.is_grasping(self.blue_sphere)
        is_green_cube_grasped = self.agent.is_grasping(self.green_cube)
        is_red_tee_grasped = self.agent.is_grasping(self.red_tee)
        is_l_shape_grasped = self.agent.is_grasping(self.l_shape)
        is_yellow_cube_grasped = self.agent.is_grasping(self.yellow_cube)
        is_red_sphere_grasped = self.agent.is_grasping(self.red_sphere)
        
        # Success: any object on bin, that object is static, and that object is not grasped
        blue_sphere_success = is_blue_sphere_on_bin & is_blue_sphere_static & (~is_blue_sphere_grasped)
        green_cube_success = is_green_cube_on_bin & is_green_cube_static & (~is_green_cube_grasped)
        red_tee_success = is_red_tee_on_bin & is_red_tee_static & (~is_red_tee_grasped)
        l_shape_success = is_l_shape_on_bin & is_l_shape_static & (~is_l_shape_grasped)
        yellow_cube_success = is_yellow_cube_on_bin & is_yellow_cube_static & (~is_yellow_cube_grasped)
        red_sphere_success = is_red_sphere_on_bin & is_red_sphere_static & (~is_red_sphere_grasped)
        
        success = torch.logical_or(
            torch.logical_or(
                torch.logical_or(blue_sphere_success, green_cube_success),
                torch.logical_or(red_tee_success, l_shape_success)
            ),
            torch.logical_or(yellow_cube_success, red_sphere_success)
        )
        
        return {
            "is_blue_sphere_grasped": is_blue_sphere_grasped,
            "is_green_cube_grasped": is_green_cube_grasped,
            "is_red_tee_grasped": is_red_tee_grasped,
            "is_l_shape_grasped": is_l_shape_grasped,
            "is_yellow_cube_grasped": is_yellow_cube_grasped,
            "is_red_sphere_grasped": is_red_sphere_grasped,
            "is_blue_sphere_on_bin": is_blue_sphere_on_bin,
            "is_green_cube_on_bin": is_green_cube_on_bin,
            "is_red_tee_on_bin": is_red_tee_on_bin,
            "is_l_shape_on_bin": is_l_shape_on_bin,
            "is_yellow_cube_on_bin": is_yellow_cube_on_bin,
            "is_red_sphere_on_bin": is_red_sphere_on_bin,
            "is_blue_sphere_static": is_blue_sphere_static,
            "is_green_cube_static": is_green_cube_static,
            "is_red_tee_static": is_red_tee_static,
            "is_l_shape_static": is_l_shape_static,
            "is_yellow_cube_static": is_yellow_cube_static,
            "is_red_sphere_static": is_red_sphere_static,
            "success": success,
        }

    def _check_object_on_bin(self, pos_object, pos_bin, object_height):
        """Check if object is on top of the bin"""
        offset = pos_object - pos_bin
        # Lenient xy check for object placement
        xy_flag = torch.linalg.norm(offset[..., :2], axis=1) <= 0.02
        z_flag = (
            torch.abs(offset[..., 2] - object_height - self.block_half_size[0]) <= 0.015
        )
        return torch.logical_and(xy_flag, z_flag)

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            is_blue_sphere_grasped=info["is_blue_sphere_grasped"],
            is_green_cube_grasped=info["is_green_cube_grasped"],
            is_red_tee_grasped=info["is_red_tee_grasped"],
            is_l_shape_grasped=info["is_l_shape_grasped"],
            is_yellow_cube_grasped=info["is_yellow_cube_grasped"],
            is_red_sphere_grasped=info["is_red_sphere_grasped"],
            tcp_pose=self.agent.tcp.pose.raw_pose,
            bin_pos=self.bin.pose.p,
        )
        if "state" in self.obs_mode:
            obs.update(
                blue_sphere_pose=self.blue_sphere.pose.raw_pose,
                green_cube_pose=self.green_cube.pose.raw_pose,
                red_tee_pose=self.red_tee.pose.raw_pose,
                l_shape_pose=self.l_shape.pose.raw_pose,
                yellow_cube_pose=self.yellow_cube.pose.raw_pose,
                red_sphere_pose=self.red_sphere.pose.raw_pose,
                tcp_to_blue_sphere_pos=self.blue_sphere.pose.p - self.agent.tcp.pose.p,
                tcp_to_green_cube_pos=self.green_cube.pose.p - self.agent.tcp.pose.p,
                tcp_to_red_tee_pos=self.red_tee.pose.p - self.agent.tcp.pose.p,
                tcp_to_l_shape_pos=self.l_shape.pose.p - self.agent.tcp.pose.p,
                tcp_to_yellow_cube_pos=self.yellow_cube.pose.p - self.agent.tcp.pose.p,
                tcp_to_red_sphere_pos=self.red_sphere.pose.p - self.agent.tcp.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # reaching reward - reward for getting close to any object
        tcp_pose = self.agent.tcp.pose.p
        blue_sphere_pos = self.blue_sphere.pose.p
        green_cube_pos = self.green_cube.pose.p
        red_tee_pos = self.red_tee.pose.p
        l_shape_pos = self.l_shape.pose.p
        yellow_cube_pos = self.yellow_cube.pose.p
        red_sphere_pos = self.red_sphere.pose.p
        
        # Find minimum distance to any object
        distances = torch.stack([
            torch.linalg.norm(tcp_pose - blue_sphere_pos, axis=1),
            torch.linalg.norm(tcp_pose - green_cube_pos, axis=1),
            torch.linalg.norm(tcp_pose - red_tee_pos, axis=1),
            torch.linalg.norm(tcp_pose - l_shape_pos, axis=1),
            torch.linalg.norm(tcp_pose - yellow_cube_pos, axis=1),
            torch.linalg.norm(tcp_pose - red_sphere_pos, axis=1),
        ], dim=0)
        obj_to_tcp_dist = torch.min(distances, dim=0)[0]
        reward = 2 * (1 - torch.tanh(5 * obj_to_tcp_dist))

        # grasp and place reward
        bin_top_pos_regular = self.bin.pose.p.clone()
        bin_top_pos_regular[:, 2] = bin_top_pos_regular[:, 2] + self.block_half_size[0] + self.object_size
        
        bin_top_pos_shapes = self.bin.pose.p.clone()
        bin_top_pos_shapes[:, 2] = bin_top_pos_shapes[:, 2] + self.block_half_size[0] + 0.007
        
        # Calculate distances to bin for each object
        blue_sphere_to_bin_dist = torch.linalg.norm(bin_top_pos_regular - blue_sphere_pos, axis=1)
        green_cube_to_bin_dist = torch.linalg.norm(bin_top_pos_regular - green_cube_pos, axis=1)
        red_tee_to_bin_dist = torch.linalg.norm(bin_top_pos_shapes - red_tee_pos, axis=1)
        l_shape_to_bin_dist = torch.linalg.norm(bin_top_pos_shapes - l_shape_pos, axis=1)
        yellow_cube_to_bin_dist = torch.linalg.norm(bin_top_pos_regular - yellow_cube_pos, axis=1)
        red_sphere_to_bin_dist = torch.linalg.norm(bin_top_pos_regular - red_sphere_pos, axis=1)
        
        # Reward based on whichever object is grasped
        grasped_flags = [
            info["is_blue_sphere_grasped"],
            info["is_green_cube_grasped"],
            info["is_red_tee_grasped"],
            info["is_l_shape_grasped"],
            info["is_yellow_cube_grasped"],
            info["is_red_sphere_grasped"],
        ]
        
        place_rewards = [
            1 - torch.tanh(5.0 * blue_sphere_to_bin_dist),
            1 - torch.tanh(5.0 * green_cube_to_bin_dist),
            1 - torch.tanh(5.0 * red_tee_to_bin_dist),
            1 - torch.tanh(5.0 * l_shape_to_bin_dist),
            1 - torch.tanh(5.0 * yellow_cube_to_bin_dist),
            1 - torch.tanh(5.0 * red_sphere_to_bin_dist),
        ]
        
        for grasped, place_reward in zip(grasped_flags, place_rewards):
            reward[grasped] = (4 + place_reward)[grasped]

        # ungrasp and static reward
        gripper_width = (self.agent.robot.get_qlimits()[0, -1, 1] * 2).to(self.device)
        is_any_object_grasped = torch.logical_or(
            torch.logical_or(
                torch.logical_or(grasped_flags[0], grasped_flags[1]),
                torch.logical_or(grasped_flags[2], grasped_flags[3])
            ),
            torch.logical_or(grasped_flags[4], grasped_flags[5])
        )
        ungrasp_reward = (
            torch.sum(self.agent.robot.get_qpos()[:, -2:], axis=1) / gripper_width
        )
        ungrasp_reward[
            ~is_any_object_grasped
        ] = 16.0  # give ungrasp a bigger reward
        
        # Static reward calculation
        objects = [self.blue_sphere, self.green_cube, self.red_tee, self.l_shape, self.yellow_cube, self.red_sphere]
        static_rewards = []
        for obj in objects:
            v = torch.linalg.norm(obj.linear_velocity, axis=1)
            av = torch.linalg.norm(obj.angular_velocity, axis=1)
            static_rewards.append(1 - torch.tanh(v * 10 + av))
        
        robot_static_reward = self.agent.is_static(0.2)
        
        # Check which objects are on bin
        on_bin_flags = [
            info["is_blue_sphere_on_bin"],
            info["is_green_cube_on_bin"],
            info["is_red_tee_on_bin"],
            info["is_l_shape_on_bin"],
            info["is_yellow_cube_on_bin"],
            info["is_red_sphere_on_bin"],
        ]
        
        is_any_on_bin = torch.logical_or(
            torch.logical_or(
                torch.logical_or(on_bin_flags[0], on_bin_flags[1]),
                torch.logical_or(on_bin_flags[2], on_bin_flags[3])
            ),
            torch.logical_or(on_bin_flags[4], on_bin_flags[5])
        )
        
        # Use static reward for whichever object is on the bin
        static_reward = static_rewards[0]  # default
        for i, on_bin in enumerate(on_bin_flags):
            static_reward = torch.where(on_bin, static_rewards[i], static_reward)
        
        reward[is_any_on_bin] = (
            6 + (ungrasp_reward + static_reward + robot_static_reward) / 3.0
        )[is_any_on_bin]

        # success reward
        reward[info["success"]] = 13
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
        # this should be equal to compute_dense_reward / max possible reward
        max_reward = 13.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward