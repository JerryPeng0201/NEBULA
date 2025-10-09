import os.path as osp
from pathlib import Path
from typing import List

import numpy as np
import sapien
import sapien.render
import torch
from transforms3d.euler import euler2quat

from nebula.core.embodiment.multi_manipulator import MultiAgent
from nebula.core.embodiment.robots.fetch import FETCH_WHEELS_COLLISION_BIT
from nebula.utils.building.ground import build_ground
from nebula.utils.scene_builder import SceneBuilder


class ColorSortingTableSceneBuilder(SceneBuilder):
    """A scene builder that creates a table with 4 colored areas for color sorting tasks"""

    def build(self):
        # Build the main table (similar to original TableSceneBuilder)
        builder = self.scene.create_actor_builder()
        model_dir = Path(osp.dirname(__file__)) / "assets"
        table_model_file = str(model_dir / "table.glb")
        scale = 1.75

        table_pose = sapien.Pose(q=euler2quat(0, 0, np.pi / 2))
        
        # Add table collision and visual
        builder.add_box_collision(
            pose=sapien.Pose(p=[0, 0, 0.9196429 / 2]),
            half_size=(2.418 / 2, 1.209 / 2, 0.9196429 / 2),
        )
        builder.add_visual_from_file(
            filename=table_model_file, scale=[scale] * 3, pose=table_pose
        )
        builder.initial_pose = sapien.Pose(
            p=[-0.12, 0, -0.9196429], q=euler2quat(0, 0, np.pi / 2)
        )
        self.table = builder.build_kinematic(name="table-workspace")

        # Define colored areas on the table surface
        # Table surface is at z=0, so colored areas will be slightly above
        area_height = 0.002  # Very thin colored areas
        area_size = 0.2  # Size of each colored area
        
        # Define 4 colors and their positions
        self.colors = {
            'red': [1.0, 0.0, 0.0, 1.0],
            'blue': [0.0, 0.0, 1.0, 1.0], 
            'green': [0.0, 1.0, 0.0, 1.0],
            'yellow': [1.0, 1.0, 0.0, 1.0]
        }
        
        # Positions for 4 colored areas (2x2 grid on table)
        area_positions = [
            [-0.15, -0.15, area_height/2],  # red area (back-left)
            [0.15, -0.15, area_height/2],   # blue area (back-right)
            [-0.15, 0.15, area_height/2],   # green area (front-left)
            [0.15, 0.15, area_height/2]     # yellow area (front-right)
        ]
        
        self.colored_areas = {}
        color_names = list(self.colors.keys())
        
        for i, (color_name, position) in enumerate(zip(color_names, area_positions)):
            area_builder = self.scene.create_actor_builder()
            area_builder.add_box_collision(
                pose=sapien.Pose(p=[0, 0, 0]),
                half_size=(area_size/2, area_size/2, area_height/2),
            )
            area_builder.add_box_visual(
                pose=sapien.Pose(p=[0, 0, 0]),
                half_size=(area_size/2, area_size/2, area_height/2),
                material=self.colors[color_name]  # Use material instead
            )
            area_builder.initial_pose = sapien.Pose(p=position)
            colored_area = area_builder.build_kinematic(name=f"{color_name}_area")
            self.colored_areas[color_name] = {
                'actor': colored_area,
                'position': np.array(position),
                'size': area_size,
                'color': self.colors[color_name]
            }

        # Build ground (same as original TableSceneBuilder)
        aabb = np.array([
            [-0.7402168, -1.2148621, -0.91964257],
            [0.4688596, 1.2030163, 3.5762787e-07],
        ])
        self.table_length = aabb[1, 0] - aabb[0, 0]
        self.table_width = aabb[1, 1] - aabb[0, 1] 
        self.table_height = aabb[1, 2] - aabb[0, 2]
        
        floor_width = 100
        if self.scene.parallel_in_single_scene:
            floor_width = 500
        self.ground = build_ground(
            self.scene, floor_width=floor_width, altitude=-self.table_height
        )
        
        # Store all scene objects
        self.scene_objects = [self.table, self.ground] + [area['actor'] for area in self.colored_areas.values()]

    def initialize(self, env_idx: torch.Tensor):
        b = len(env_idx)
        
        # Initialize table pose
        self.table.set_pose(
            sapien.Pose(p=[-0.12, 0, -0.9196429], q=euler2quat(0, 0, np.pi / 2))
        )
        
        # Initialize colored areas
        area_positions = [
            [-0.15, -0.15, 0.001],  # red area
            [0.15, -0.15, 0.001],   # blue area  
            [-0.15, 0.15, 0.001],   # green area
            [0.15, 0.15, 0.001]     # yellow area
        ]
        
        color_names = list(self.colored_areas.keys())
        for i, color_name in enumerate(color_names):
            self.colored_areas[color_name]['actor'].set_pose(
                sapien.Pose(p=area_positions[i])
            )
        
        # Initialize robots (copied from original TableSceneBuilder)
        if self.env.robot_uids == "panda":
            qpos = np.array([
                0.0, np.pi / 8, 0, -np.pi * 5 / 8, 0, np.pi * 3 / 4, np.pi / 4, 0.04, 0.04,
            ])
            if self.env._enhanced_determinism:
                qpos = (
                    self.env._batched_episode_rng[env_idx].normal(
                        0, self.robot_init_qpos_noise, len(qpos)
                    )
                    + qpos
                )
            else:
                qpos = (
                    self.env._episode_rng.normal(
                        0, self.robot_init_qpos_noise, (b, len(qpos))
                    )
                    + qpos
                )
            qpos[:, -2:] = 0.04
            self.env.agent.reset(qpos)
            self.env.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))
        elif self.env.robot_uids == "panda_wristcam":
            qpos = np.array([
                0.0, np.pi / 8, 0, -np.pi * 5 / 8, 0, np.pi * 3 / 4, -np.pi / 4, 0.04, 0.04
            ])
            if self.env._enhanced_determinism:
                qpos = (
                    self.env._batched_episode_rng[env_idx].normal(
                        0, self.robot_init_qpos_noise, len(qpos)
                    )
                    + qpos
                )
            else:
                qpos = (
                    self.env._episode_rng.normal(
                        0, self.robot_init_qpos_noise, (b, len(qpos))
                    )
                    + qpos
                )
            qpos[:, -2:] = 0.04
            self.env.agent.reset(qpos)
            self.env.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))
        elif self.env.robot_uids in [
            "xarm6_allegro_left",
            "xarm6_allegro_right", 
            "xarm6_robotiq",
            "xarm6_nogripper",
        ]:
            qpos = self.env.agent.keyframes["rest"].qpos
            qpos = (
                self.env._episode_rng.normal(
                    0, self.robot_init_qpos_noise, (b, len(qpos))
                )
                + qpos
            )
            self.env.agent.reset(qpos)
            self.env.agent.robot.set_pose(sapien.Pose([-0.522, 0, 0]))
        elif self.env.robot_uids == "fetch":
            qpos = np.array([
                0, 0, 0, 0.386, 0, 0, 0, -np.pi / 4, 0, np.pi / 4, 0, np.pi / 3, 0, 0.015, 0.015,
            ])
            self.env.agent.reset(qpos)
            self.env.agent.robot.set_pose(sapien.Pose([-1.05, 0, -self.table_height]))

            self.ground.set_collision_group_bit(
                group=2, bit_idx=FETCH_WHEELS_COLLISION_BIT, bit=1
            )
        elif self.env.robot_uids == ("panda", "panda"):
            agent: MultiAgent = self.env.agent
            qpos = np.array([
                0.0, np.pi / 8, 0, -np.pi * 5 / 8, 0, np.pi * 3 / 4, np.pi / 4, 0.04, 0.04,
            ])
            if self.env._enhanced_determinism:
                qpos = (
                    self.env._batched_episode_rng[env_idx].normal(
                        0, self.robot_init_qpos_noise, len(qpos)
                    )
                    + qpos
                )
            else:
                qpos = (
                    self.env._episode_rng.normal(
                        0, self.robot_init_qpos_noise, (b, len(qpos))
                    )
                    + qpos
                )
            qpos[:, -2:] = 0.04
            agent.agents[1].reset(qpos)
            agent.agents[1].robot.set_pose(
                sapien.Pose([0, 0.75, 0], q=euler2quat(0, 0, -np.pi / 2))
            )
            agent.agents[0].reset(qpos)
            agent.agents[0].robot.set_pose(
                sapien.Pose([0, -0.75, 0], q=euler2quat(0, 0, np.pi / 2))
            )
        elif self.env.robot_uids == ("panda_wristcam", "panda_wristcam"):
            agent: MultiAgent = self.env.agent
            qpos = np.array([
                0.0, np.pi / 8, 0, -np.pi * 5 / 8, 0, np.pi * 3 / 4, np.pi / 4, 0.04, 0.04,
            ])
            if self.env._enhanced_determinism:
                qpos = (
                    self.env._batched_episode_rng[env_idx].normal(
                        0, self.robot_init_qpos_noise, len(qpos)
                    )
                    + qpos
                )
            else:
                qpos = (
                    self.env._episode_rng.normal(
                        0, self.robot_init_qpos_noise, (b, len(qpos))
                    )
                    + qpos
                )
            qpos[:, -2:] = 0.04
            agent.agents[1].reset(qpos)
            agent.agents[1].robot.set_pose(
                sapien.Pose([0, 0.75, 0], q=euler2quat(0, 0, -np.pi / 2))
            )
            agent.agents[0].reset(qpos)
            agent.agents[0].robot.set_pose(
                sapien.Pose([0, -0.75, 0], q=euler2quat(0, 0, np.pi / 2))
            )
        elif (
            "dclaw" in self.env.robot_uids
            or "allegro" in self.env.robot_uids
            or "trifinger" in self.env.robot_uids
        ):
            # Need to specify the robot qpos for each sub-scenes using tensor api
            pass
        elif self.env.robot_uids == "panda_stick":
            qpos = np.array([
                0.0, np.pi / 8, 0, -np.pi * 5 / 8, 0, np.pi * 3 / 4, np.pi / 4,
            ])
            if self.env._enhanced_determinism:
                qpos = (
                    self.env._batched_episode_rng[env_idx].normal(
                        0, self.robot_init_qpos_noise, len(qpos)
                    )
                    + qpos
                )
            else:
                qpos = (
                    self.env._episode_rng.normal(
                        0, self.robot_init_qpos_noise, (b, len(qpos))
                    )
                    + qpos
                )
            self.env.agent.reset(qpos)
            self.env.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))
        elif self.env.robot_uids in ["widowxai", "widowxai_wristcam"]:
            qpos = self.env.agent.keyframes["ready_to_grasp"].qpos
            self.env.agent.reset(qpos)
        elif self.env.robot_uids == "so100":
            qpos = np.array([0, np.pi / 2, np.pi / 2, np.pi / 2, -np.pi / 2, 1.0])
            qpos = (
                self.env._episode_rng.normal(
                    0, self.robot_init_qpos_noise, (b, len(qpos))
                )
                + qpos
            )
            self.env.agent.reset(qpos)
            self.env.agent.robot.set_pose(
                sapien.Pose([-0.725, 0, 0], q=euler2quat(0, 0, np.pi / 2))
            )