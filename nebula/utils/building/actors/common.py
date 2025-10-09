"""
Common utilities for adding primitive prebuilt shapes to a scene
"""

from typing import Optional, Union

import numpy as np
import sapien
import sapien.render

from nebula.core.simulation.scene import NEBULAScene
from nebula.utils.building.actor_builder import ActorBuilder
from nebula.utils.structs.pose import Pose
from nebula.utils.structs.types import Array


def _build_by_type(
    builder: ActorBuilder,
    name,
    body_type,
    scene_idxs: Optional[Array] = None,
    initial_pose: Optional[Union[Pose, sapien.Pose]] = None,
):
    if scene_idxs is not None:
        builder.set_scene_idxs(scene_idxs)
    if initial_pose is not None:
        builder.set_initial_pose(initial_pose)
    if body_type == "dynamic":
        actor = builder.build(name=name)
    elif body_type == "static":
        actor = builder.build_static(name=name)
    elif body_type == "kinematic":
        actor = builder.build_kinematic(name=name)
    else:
        raise ValueError(f"Unknown body type {body_type}")
    return actor


# Primitive Shapes
def build_cube(
    scene: NEBULAScene,
    half_size: float,
    color,
    name: str,
    body_type: str = "dynamic",
    add_collision: bool = True,
    scene_idxs: Optional[Array] = None,
    initial_pose: Optional[Union[Pose, sapien.Pose]] = None,
):
    builder = scene.create_actor_builder()
    if add_collision:
        builder.add_box_collision(
            half_size=[half_size] * 3,
        )
    builder.add_box_visual(
        half_size=[half_size] * 3,
        material=sapien.render.RenderMaterial(
            base_color=color,
        ),
    )
    return _build_by_type(builder, name, body_type, scene_idxs, initial_pose)


def build_box(
    scene: NEBULAScene,
    half_sizes,
    color,
    name: str,
    body_type: str = "dynamic",
    add_collision: bool = True,
    scene_idxs: Optional[Array] = None,
    initial_pose: Optional[Union[Pose, sapien.Pose]] = None,
):
    builder = scene.create_actor_builder()
    if add_collision:
        builder.add_box_collision(
            half_size=half_sizes,
        )
    builder.add_box_visual(
        half_size=half_sizes,
        material=sapien.render.RenderMaterial(
            base_color=color,
        ),
    )
    return _build_by_type(builder, name, body_type, scene_idxs, initial_pose)

def build_target_box(
    scene: NEBULAScene,
    half_sizes,
    color,
    name: str,
    body_type: str = "kinematic",
    add_collision: bool = False,
    scene_idxs: Optional[Array] = None,
    initial_pose: Optional[Union[Pose, sapien.Pose]] = None,
):
    builder = scene.create_actor_builder()
    
    # Make color transparent
    transparent_color = [color[0], color[1], color[2], 1]  
    
    # Create wireframe using thin capsules for each edge
    edge_radius = 0.001
    segments_per_edge = 6  # Number of dotted segments per edge
    gap_ratio = 0.3  # 30% gap between segments
    
    # Define 12 edges of the box
    edges = [
        # Bottom face
        ([-half_sizes[0], -half_sizes[1], -half_sizes[2]], [half_sizes[0], -half_sizes[1], -half_sizes[2]]),
        ([half_sizes[0], -half_sizes[1], -half_sizes[2]], [half_sizes[0], half_sizes[1], -half_sizes[2]]),
        ([half_sizes[0], half_sizes[1], -half_sizes[2]], [-half_sizes[0], half_sizes[1], -half_sizes[2]]),
        ([-half_sizes[0], half_sizes[1], -half_sizes[2]], [-half_sizes[0], -half_sizes[1], -half_sizes[2]]),
        # Top face
        ([-half_sizes[0], -half_sizes[1], half_sizes[2]], [half_sizes[0], -half_sizes[1], half_sizes[2]]),
        ([half_sizes[0], -half_sizes[1], half_sizes[2]], [half_sizes[0], half_sizes[1], half_sizes[2]]),
        ([half_sizes[0], half_sizes[1], half_sizes[2]], [-half_sizes[0], half_sizes[1], half_sizes[2]]),
        ([-half_sizes[0], half_sizes[1], half_sizes[2]], [-half_sizes[0], -half_sizes[1], half_sizes[2]]),
        # Vertical edges
        ([-half_sizes[0], -half_sizes[1], -half_sizes[2]], [-half_sizes[0], -half_sizes[1], half_sizes[2]]),
        ([half_sizes[0], -half_sizes[1], -half_sizes[2]], [half_sizes[0], -half_sizes[1], half_sizes[2]]),
        ([half_sizes[0], half_sizes[1], -half_sizes[2]], [half_sizes[0], half_sizes[1], half_sizes[2]]),
        ([-half_sizes[0], half_sizes[1], -half_sizes[2]], [-half_sizes[0], half_sizes[1], half_sizes[2]]),
    ]
    
    for start, end in edges:
        start, end = np.array(start), np.array(end)
        edge_vec = end - start
        edge_length = np.linalg.norm(edge_vec)
        edge_dir = edge_vec / edge_length
        
        # Create dotted segments
        segment_length = edge_length / segments_per_edge * (1 - gap_ratio)
        segment_spacing = edge_length / segments_per_edge
        
        for i in range(segments_per_edge):
            segment_center = start + edge_dir * (i * segment_spacing + segment_length / 2)
            
            # Determine orientation based on edge direction
            if abs(edge_dir[0]) > 0.9:  # X-aligned
                quat = [0.707, 0, 0, 0.707]
            elif abs(edge_dir[1]) > 0.9:  # Y-aligned
                quat = [0.707, 0, 0.707, 0]
            else:  # Z-aligned
                quat = [1, 0, 0, 0]
            
            builder.add_capsule_visual(
                sapien.Pose(p=segment_center, q=quat),
                edge_radius,
                segment_length / 2,
                material=sapien.render.RenderMaterial(base_color=transparent_color),
            )
    
    return _build_by_type(builder, name, body_type, scene_idxs, initial_pose)


def build_cylinder(
    scene: NEBULAScene,
    radius: float,
    half_length: float,
    color,
    name: str,
    body_type: str = "dynamic",
    add_collision: bool = True,
    scene_idxs: Optional[Array] = None,
    initial_pose: Optional[Union[Pose, sapien.Pose]] = None,
):
    builder = scene.create_actor_builder()
    if add_collision:
        builder.add_cylinder_collision(
            radius=radius,
            half_length=half_length,
        )
    builder.add_cylinder_visual(
        radius=radius,
        half_length=half_length,
        material=sapien.render.RenderMaterial(
            base_color=color,
        ),
    )
    return _build_by_type(builder, name, body_type, scene_idxs, initial_pose)


def build_sphere(
    scene: NEBULAScene,
    radius: float,
    color,
    name: str,
    body_type: str = "dynamic",
    add_collision: bool = True,
    scene_idxs: Optional[Array] = None,
    initial_pose: Optional[Union[Pose, sapien.Pose]] = None,
):
    builder = scene.create_actor_builder()
    if add_collision:
        builder.add_sphere_collision(
            radius=radius,
        )
    builder.add_sphere_visual(
        radius=radius,
        material=sapien.render.RenderMaterial(
            base_color=color,
        ),
    )
    return _build_by_type(builder, name, body_type, scene_idxs, initial_pose)


def build_red_white_target(
    scene: NEBULAScene,
    radius: float,
    thickness: float,
    name: str,
    body_type: str = "dynamic",
    add_collision: bool = True,
    scene_idxs: Optional[Array] = None,
    initial_pose: Optional[Union[Pose, sapien.Pose]] = None,
):
    TARGET_RED = np.array([194, 19, 22, 255]) / 255
    builder = scene.create_actor_builder()
    builder.add_cylinder_visual(
        radius=radius,
        half_length=thickness / 2,
        material=sapien.render.RenderMaterial(base_color=TARGET_RED),
    )
    builder.add_cylinder_visual(
        radius=radius * 4 / 5,
        half_length=thickness / 2 + 1e-5,
        material=sapien.render.RenderMaterial(base_color=[1, 1, 1, 1]),
    )
    builder.add_cylinder_visual(
        radius=radius * 3 / 5,
        half_length=thickness / 2 + 2e-5,
        material=sapien.render.RenderMaterial(base_color=TARGET_RED),
    )
    builder.add_cylinder_visual(
        radius=radius * 2 / 5,
        half_length=thickness / 2 + 3e-5,
        material=sapien.render.RenderMaterial(base_color=[1, 1, 1, 1]),
    )
    builder.add_cylinder_visual(
        radius=radius * 1 / 5,
        half_length=thickness / 2 + 4e-5,
        material=sapien.render.RenderMaterial(base_color=TARGET_RED),
    )
    if add_collision:
        builder.add_cylinder_collision(
            radius=radius,
            half_length=thickness / 2,
        )
        builder.add_cylinder_collision(
            radius=radius * 4 / 5,
            half_length=thickness / 2 + 1e-5,
        )
        builder.add_cylinder_collision(
            radius=radius * 3 / 5,
            half_length=thickness / 2 + 2e-5,
        )
        builder.add_cylinder_collision(
            radius=radius * 2 / 5,
            half_length=thickness / 2 + 3e-5,
        )
        builder.add_cylinder_collision(
            radius=radius * 1 / 5,
            half_length=thickness / 2 + 4e-5,
        )
    return _build_by_type(builder, name, body_type, scene_idxs, initial_pose)


def build_twocolor_peg(
    scene: NEBULAScene,
    length,
    width,
    color_1,
    color_2,
    name: str,
    body_type="dynamic",
    add_collision: bool = True,
    scene_idxs: Optional[Array] = None,
    initial_pose: Optional[Union[Pose, sapien.Pose]] = None,
):
    builder = scene.create_actor_builder()
    if add_collision:
        builder.add_box_collision(
            half_size=[length, width, width],
        )
    builder.add_box_visual(
        pose=sapien.Pose(p=[-length / 2, 0, 0]),
        half_size=[length / 2, width, width],
        material=sapien.render.RenderMaterial(
            base_color=color_1,
        ),
    )
    builder.add_box_visual(
        pose=sapien.Pose(p=[length / 2, 0, 0]),
        half_size=[length / 2, width, width],
        material=sapien.render.RenderMaterial(
            base_color=color_2,
        ),
    )
    return _build_by_type(builder, name, body_type, scene_idxs, initial_pose)


RED_COLOR = [220 / 255, 12 / 255, 12 / 255, 1]
BLUE_COLOR = [0 / 255, 44 / 255, 193 / 255, 1]
GREEN_COLOR = [17 / 255, 190 / 255, 70 / 255, 1]


def build_fourcolor_peg(
    scene: NEBULAScene,
    length,
    width,
    name: str,
    color_1=RED_COLOR,
    color_2=BLUE_COLOR,
    color_3=GREEN_COLOR,
    color_4=[1, 1, 1, 1],
    body_type="dynamic",
    add_collision: bool = True,
    scene_idxs: Optional[Array] = None,
    initial_pose: Optional[Union[Pose, sapien.Pose]] = None,
):
    """
    A peg with four sections and four different colors. Useful for visualizing every possible rotation without any symmetries
    """
    builder = scene.create_actor_builder()
    if add_collision:
        builder.add_box_collision(
            half_size=[length, width, width],
        )
    builder.add_box_visual(
        pose=sapien.Pose(p=[-length / 2, -width / 2, 0]),
        half_size=[length / 2, width / 2, width],
        material=sapien.render.RenderMaterial(
            base_color=color_1,
        ),
    )
    builder.add_box_visual(
        pose=sapien.Pose(p=[length / 2, -width / 2, 0]),
        half_size=[length / 2, width / 2, width],
        material=sapien.render.RenderMaterial(
            base_color=color_2,
        ),
    )
    builder.add_box_visual(
        pose=sapien.Pose(p=[-length / 2, width / 2, 0]),
        half_size=[length / 2, width / 2, width],
        material=sapien.render.RenderMaterial(
            base_color=color_3,
        ),
    )
    builder.add_box_visual(
        pose=sapien.Pose(p=[length / 2, width / 2, 0]),
        half_size=[length / 2, width / 2, width],
        material=sapien.render.RenderMaterial(
            base_color=color_4,
        ),
    )
    return _build_by_type(builder, name, body_type, scene_idxs, initial_pose)


def build_colorful_cube(
    scene: NEBULAScene,
    half_size: float,
    color,
    name: str,
    body_type: str = "dynamic",
    add_collision: bool = True,
    scene_idxs: Optional[Array] = None,
    initial_pose: Optional[Union[Pose, sapien.Pose]] = None,
):
    builder = scene.create_actor_builder()

    if add_collision:
        builder._mass = 0.1
        cube_material = sapien.pysapien.physx.PhysxMaterial(
            static_friction=5, dynamic_friction=3, restitution=0
        )
        builder.add_box_collision(
            half_size=[half_size] * 3,
            material=cube_material,
        )
    builder.add_box_visual(
        half_size=[half_size] * 3,
        material=sapien.render.RenderMaterial(
            base_color=color,
        ),
    )
    return _build_by_type(builder, name, body_type, scene_idxs, initial_pose)
