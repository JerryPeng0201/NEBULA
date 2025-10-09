from typing import Any, Dict, Union, List
import numpy as np
import sapien
import torch
import random

import nebula.core.simulation.utils.randomization as randomization
from nebula.core.embodiment.robots import Fetch, Panda
from nebula.core.simulation.engine import BaseEnv
from nebula.core.sensors.camera import CameraConfig
from nebula.utils import sapien_utils
from nebula.utils.building import actors
from nebula.utils.registration import register_env
from nebula.utils.scene_builder.table import TableSceneBuilder
from nebula.utils.structs.pose import Pose
from nebula.utils.structs.types import GPUMemoryConfig, SimConfig

LANGUAGE_HARD_DOC_STRING = """Task Description:
Test language logic understanding and conditional branch execution capabilities:
1. Conditional logic: "If the green cube is smaller than the red one, pick it up. Otherwise, take the red one"
2. Sequential multi-step: "First pick the red cube, then place it next to the blue one"
3. Reference-based selection: "Pick the one that matches the color of the object near the corner"

Language Commands:
- Task Set #1: Conditional commands - Size comparisons with if-then-else logic
- Task Set #2: Sequential commands - Multi-step tasks requiring specific order
- Task Set #3: Reference commands - Selection based on spatial relationships

Randomizations:
- Object colors, sizes, and positions are randomized
- Reference object positions vary
- Task conditions change dynamically
- Multiple distractors present

Success Conditions:
- Correct execution of conditional logic
- Proper sequence of actions for multi-step tasks
- Accurate reference-based object selection
- Task-specific placement or manipulation goals achieved
"""

class LanguageHardPickCubeBaseEnv(BaseEnv):
    
    SUPPORTED_ROBOTS = ["panda", "fetch"]
    agent: Union[Panda, Fetch]
    
    lift_thresh = 0.05
    place_thresh = 0.08
    
    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        
        # Object colors
        self.object_colors = {
            "red": [1, 0, 0, 1],
            "blue": [0, 0, 1, 1], 
            "green": [0, 1, 0, 1],
            "yellow": [1, 1, 0, 1],
            "purple": [1, 0, 1, 1],
            "white": [1, 1, 1, 1],
            "orange": [1, 0.5, 0, 1],
            "black": [0, 0, 0, 1],
            "gray": [0.5, 0.5, 0.5, 1],
            "pink": [1, 0.75, 0.8, 1],
            "brown": [0.6, 0.3, 0, 1],
        }
        
        # Object sizes with more variation for conditional logic
        base_sizes = {"small": 0.010, "medium": 0.015, "large": 0.020}
        self.object_sizes = {}
        for size_name, base_size in base_sizes.items():
            random_factor = 1.0 + (random.random() - 0.5) * 0.1
            self.object_sizes[size_name] = base_size * random_factor
            
        # Task types for hard difficulty
        self.task_types = ["conditional", "sequential", "reference"]
        
        # Language templates for each task type
        self.command_templates = {
            "conditional": [
                "If the {color1} cube is smaller than the {color2} one, pick the {color1} cube. Otherwise, pick the {color2} cube",
                "When the {color1} cube is larger than the {color2} one, pick the {color1} cube. Otherwise, pick the {color2} cube",
            
                "Pick the {color1} cube if it exists, otherwise pick the {color2} cube",
                "If there's a {color1} cube present, pick it. Otherwise, pick the {color2} cube"
            ],
            "sequential": [
                "First pick the {color1} cube, then place it next to the {color2} one",
                "Pick the {color1} cube and place it beside the {color2} cube"
            ],
            "reference": [
                "Pick the cube that has the same color as the largest cube",
                "Pick the cube that has the same size as the {color1} cube",
                "Pick the smallest cube that is not {color1}",
                "Pick the cube with the same color as the smallest cube"
            ]
        }
        
        self.task_instruction = ""
        self.execution_state = {"phase": "initial", "picked_objects": [], "target_sequence": []}
        
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
        pose = sapien_utils.look_at(eye=[0.6, 0.6, 0.8], target=[0, 0, 0.1])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)
    
    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18
            )
        )

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        if not hasattr(self, 'table_scene'):
            self.table_scene = TableSceneBuilder(self, robot_init_qpos_noise=self.robot_init_qpos_noise)
            self.table_scene.build()
        
        if not hasattr(self, 'objects'):
            self.objects = {}
            
            base_sizes = {
                "small": 0.010,
                "medium": 0.018,
                "large": 0.026
            }
            
            all_colors = list(self.object_colors.keys())
            selected_colors = random.sample(all_colors, 6)
            
            object_configs = []
            
            color_A = selected_colors[0]
            object_configs.append({
                "color": color_A,
                "size": "large",
                "size_value": base_sizes["large"] + 0.004
            })
            object_configs.append({
                "color": color_A,
                "size": "medium",
                "size_value": base_sizes["medium"]
            })
            
            color_B = selected_colors[1]
            object_configs.append({
                "color": color_B,
                "size": "small",
                "size_value": base_sizes["small"] - 0.003
            })
            object_configs.append({
                "color": color_B,
                "size": "medium",
                "size_value": base_sizes["medium"] + 0.001
            })
            
            color_C = selected_colors[2]
            color_D = selected_colors[3]
            shared_size = base_sizes["small"] + 0.001
            
            object_configs.append({
                "color": color_C,
                "size": "small",
                "size_value": shared_size
            })
            object_configs.append({
                "color": color_D,
                "size": "small",
                "size_value": shared_size
            })
            
            color_E = selected_colors[4]
            object_configs.append({
                "color": color_E,
                "size": "large",
                "size_value": base_sizes["large"] - 0.001
            })
            
            random.shuffle(object_configs)
            
            # Create objects
            for i, config in enumerate(object_configs):
                obj_name = f"cube{i}"
                
                builder = self.scene.create_actor_builder()
                builder.add_box_collision(half_size=[config["size_value"]] * 3)
                
                material = sapien.render.RenderMaterial()
                material.base_color = self.object_colors[config["color"]]
                builder.add_box_visual(half_size=[config["size_value"]] * 3, material=material)
                
                builder.initial_pose = sapien.Pose(p=[0, 0, 1.0])
                actor = builder.build(name=obj_name)
                
                self.objects[obj_name] = {
                    "actor": actor,
                    "color": config["color"],
                    "size": config["size"],
                    "size_value": config["size_value"],
                    "index": i
                }

    def _generate_task_instruction(self):
        colors_available = list(set(obj["color"] for obj in self.objects.values()))
        
        if self.current_task_type == "conditional":
            # Count number and size of objects for each color
            color_counts = {}
            color_objects = {}
            for obj in self.objects.values():
                color = obj["color"]
                color_counts[color] = color_counts.get(color, 0) + 1
                if color not in color_objects:
                    color_objects[color] = []
                color_objects[color].append(obj)
            
            # Select task template type
            template_type = random.choice(["size_comparison", "presence_check"])
            
            if template_type == "size_comparison":
                # Define minimum visible difference threshold (at least 3mm difference for visual distinction)
                MIN_SIZE_DIFF = 0.003
                
                # Find suitable color pairs for size comparison
                suitable_pairs = []
                
                # Prioritize colors with single objects (avoid ambiguity)
                unique_colors = [color for color, count in color_counts.items() if count == 1]
                
                # Find color pairs with clear size differences from unique colors
                for i, color1 in enumerate(unique_colors):
                    for color2 in unique_colors[i+1:]:
                        obj1 = color_objects[color1][0]
                        obj2 = color_objects[color2][0]
                        size_diff = abs(obj1["size_value"] - obj2["size_value"])
                        
                        if size_diff >= MIN_SIZE_DIFF:
                            suitable_pairs.append((color1, color2, size_diff))
                
                # If no suitable unique color pairs found, consider non-unique colors
                if not suitable_pairs:
                    for color1 in colors_available:
                        for color2 in colors_available:
                            if color1 != color2:
                                obj1 = color_objects[color1][0] 
                                obj2 = color_objects[color2][0]
                                size_diff = abs(obj1["size_value"] - obj2["size_value"])
                                
                                if size_diff >= MIN_SIZE_DIFF:
                                    suitable_pairs.append((color1, color2, size_diff))
                
                if suitable_pairs:
                    # Prioritize pairs with more obvious differences
                    suitable_pairs.sort(key=lambda x: x[2], reverse=True)
                    # Randomly select from top most distinct pairs
                    top_pairs = suitable_pairs[:min(5, len(suitable_pairs))]
                    color1, color2, _ = random.choice(top_pairs)
                else:
                    # If still no suitable pairs, select the most different objects
                    all_objs = list(self.objects.values())
                    all_objs.sort(key=lambda x: x["size_value"])
                    
                    if len(all_objs) >= 2:
                        smallest_obj = all_objs[0]
                        largest_obj = all_objs[-1]
                        color1 = smallest_obj["color"]
                        color2 = largest_obj["color"]
                        
                        if color1 == color2 and len(all_objs) > 2:
                            color2 = all_objs[-2]["color"]
                    else:
                        color1 = colors_available[0]
                        color2 = colors_available[1] if len(colors_available) > 1 else colors_available[0]
                
                template = random.choice([
                    "If the {color1} cube is smaller than the {color2} one, pick the {color1} cube. Otherwise, pick the {color2} cube",
                    "When the {color1} cube is larger than the {color2} one, pick the {color1} cube. Otherwise, pick the {color2} cube"
                ])
                
            else:  # presence_check
                # Choose colors for "present" condition
                all_possible_colors = ["red", "blue", "green", "yellow", "purple", "orange"]
                non_existing_colors = [c for c in all_possible_colors if c not in colors_available]
                
                if random.random() < 0.5 and non_existing_colors:
                    color1 = random.choice(non_existing_colors)
                    color2 = random.choice(colors_available)
                else:
                    if len(colors_available) >= 2:
                        color1, color2 = random.sample(colors_available, 2)
                    else:
                        color1 = colors_available[0]
                        color2 = colors_available[0] if len(colors_available) == 1 else "blue"
                
                template = random.choice([
                    "Pick the {color1} cube if it exists, otherwise pick the {color2} cube",
                    "If there's a {color1} cube present, pick it. Otherwise, pick the {color2} cube"
                ])
            
            self.task_instruction = template.format(color1=color1, color2=color2)
            self.target_criteria = {
                "type": "conditional",
                "color1": color1,
                "color2": color2,
                "condition": "size_comparison" if template_type == "size_comparison" else "presence_check"
            }
            
        elif self.current_task_type == "sequential":
            # Any two colors work since each color has only one object
            color1, color2 = random.sample(colors_available, 2)
            
            template = random.choice(self.command_templates["sequential"])
            self.task_instruction = template.format(color1=color1, color2=color2)
            
            self.target_criteria = {
                "type": "sequential",
                "color1": color1,
                "color2": color2,
                "sequence": ["pick_color1", "place_next_to_color2"]
            }
        
        else:  # reference
            # Find object pairs with matching sizes
            size_groups = {}
            for obj in self.objects.values():
                size_val = round(obj["size_value"], 6)
                if size_val not in size_groups:
                    size_groups[size_val] = []
                size_groups[size_val].append(obj)
            
            shared_sizes = [objs for objs in size_groups.values() if len(objs) == 2]
            
            smallest_obj = min(self.objects.values(), key=lambda x: x["size_value"])
            smallest_color = smallest_obj["color"]
            
            reference_templates = []
            
            # Only add feasible templates
            reference_templates.append("Pick the cube that has the same color as the largest cube")
            reference_templates.append("Pick the cube with the same color as the smallest cube")
            
            if shared_sizes:
                reference_templates.append("Pick the cube that has the same size as the {color1} cube")
            
            # Only generate "not that color" task when smallest object color is determined
            # 20% chance to generate this type of task
            if random.random() < 0.2:
                self.task_instruction = f"Pick the smallest cube that is not {smallest_color}"
                self.target_criteria = {
                    "type": "reference",
                    "excluded_color": smallest_color,
                    "reference_type": "smallest_not_color"
                }
                return
            
            template = random.choice(reference_templates)
            
            if "{color1}" in template:
                if "same size" in template and shared_sizes:
                    color1 = random.choice(shared_sizes[0])["color"]
                else:
                    color1 = random.choice(colors_available)
                    
                self.task_instruction = template.format(color1=color1)
                
                if "same size" in template:
                    self.target_criteria = {
                        "type": "reference",
                        "reference_color": color1,
                        "reference_type": "size_match"
                    }
            else:
                self.task_instruction = template
                if "largest" in template:
                    self.target_criteria = {
                        "type": "reference",
                        "reference_type": "same_as_largest"
                    }
                elif "smallest" in template:
                    self.target_criteria = {
                        "type": "reference",
                        "reference_type": "same_as_smallest"
                    }

    def _generate_safe_positions(
        self,
        num_objects: int,
        min_clearance: float = 0.08,
        edge_margin: float = 0.02, 
        r_min: float = 0.08,  
        r_max: float = 0.38, 
        max_trials: int = 2000,
    ):
        rng = np.random.default_rng()
        base_ws = {"x_range": [-0.18, 0.18], "y_range": [-0.18, 0.18], "center": [0.00, 0.00]}
        x_lo = base_ws["x_range"][0] + edge_margin
        x_hi = base_ws["x_range"][1] - edge_margin
        y_lo = base_ws["y_range"][0] + edge_margin
        y_hi = base_ws["y_range"][1] - edge_margin
        cx, cy = base_ws["center"]

        # Collect radii for all objects
        radii = []
        obj_list = list(self.objects.items())[:num_objects]
        for _, obj_data in obj_list:
            # Cube bounding circle radius
            r_obj = float(obj_data["size_value"]) * 1.414 + 0.003
            radii.append(r_obj)
        radii = np.asarray(radii, dtype=np.float32)

        def is_reachable(x, y):
            rx, ry = x - cx, y - cy
            r = float(np.hypot(rx, ry))
            return (r_min <= r <= r_max)

        def in_bounds(x, y):
            return (x_lo <= x <= x_hi) and (y_lo <= y <= y_hi)

        def far_enough(new_pt, pts, idx):
            if not pts:
                return True
            x, y = new_pt
            for j, (px, py) in enumerate(pts):
                d = float(np.hypot(px - x, py - y))
                min_dist = radii[j] + radii[idx] + min_clearance
                if d < min_dist:
                    return False
            return True

        placed = []
        trials = 0
        
        # Try random placement
        while len(placed) < num_objects and trials < max_trials:
            trials += 1
            x = rng.uniform(x_lo, x_hi)
            y = rng.uniform(y_lo, y_hi)
            
            idx = len(placed)
            if in_bounds(x, y) and is_reachable(x, y) and far_enough((x, y), placed, idx):
                placed.append((x, y))

        if len(placed) == num_objects:
            return [list(p) for p in placed]
        
        # Predefined safe positions (6-8 objects)
        fallback_positions = [
            [0.10, 0.10],   # Top right
            [-0.10, 0.10],  # Top left  
            [0.10, -0.10],  # Bottom right
            [-0.10, -0.10], # Bottom left
            [0.15, 0.0],    # Center right
            [-0.15, 0.0],   # Center left
            [0.0, 0.15],    # Center top
            [0.0, -0.15],   # Center bottom
        ]
        
        # Ensure all positions are reachable
        valid_fallback = []
        for pos in fallback_positions:
            if is_reachable(pos[0], pos[1]):
                valid_fallback.append(pos)
        
        # If not enough positions, generate more
        if len(valid_fallback) < num_objects:
            # Distribute uniformly in circular area
            angle_step = 2 * np.pi / num_objects
            radius = (r_min + r_max) / 2
            for i in range(num_objects):
                angle = i * angle_step
                x = cx + radius * np.cos(angle)
                y = cy + radius * np.sin(angle)
                if is_reachable(x, y):
                    valid_fallback.append([x, y])
        
        # Select first num_objects positions
        out = []
        for i in range(num_objects):
            if i < len(valid_fallback):
                pos = valid_fallback[i]
                # Add small random perturbation
                x = pos[0] + (rng.random() - 0.5) * 0.02
                y = pos[1] + (rng.random() - 0.5) * 0.02
                x = float(np.clip(x, x_lo, x_hi))
                y = float(np.clip(y, y_lo, y_hi))
                out.append([x, y])
            else:
                # If still not enough, use variants of existing positions
                base_pos = out[i % len(out)]
                x = base_pos[0] + (rng.random() - 0.5) * 0.05
                y = base_pos[1] + (rng.random() - 0.5) * 0.05
                x = float(np.clip(x, x_lo, x_hi))
                y = float(np.clip(y, y_lo, y_hi))
                out.append([x, y])
        
        return out

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            
            # Reset execution state
            self.execution_state = {"phase": "initial", "picked_objects": [], "target_sequence": []}
            
            # Initialize target_criteria to avoid AttributeError
            self.target_criteria = {"type": "conditional"}
            
            # Randomly select task type
            self.current_task_type = random.choice(self.task_types)
            self._generate_task_instruction()
            
            # Generate safe positions for objects
            safe_positions = self._generate_safe_positions(num_objects=len(self.objects))
            
            # Place one object near corner for reference tasks
            if self.current_task_type == "reference":
                corner_idx = random.randint(0, len(self.objects) - 1)
                corner_position = [0.15, 0.15]  # Near corner
                safe_positions[corner_idx] = corner_position
                self.reference_object_idx = corner_idx
            
            for i, (obj_name, obj_data) in enumerate(self.objects.items()):
                obj_xyz = torch.zeros((b, 3))
                obj_xyz[:, :2] = torch.tensor(safe_positions[i]) + (torch.rand((b, 2)) * 2 - 1) * 0.01
                obj_xyz[:, 2] = obj_data["size_value"]
                obj_data["actor"].set_pose(Pose.create_from_pq(obj_xyz))

    def _evaluate_condition(self):
        """Evaluate conditional logic for conditional tasks
        Returns: (condition_met, target_object)
        - condition_met: True if condition is satisfied, False otherwise
        - target_object: the object to pick based on condition evaluation
        """
        if self.target_criteria["type"] != "conditional":
            return None
            
        color1 = self.target_criteria["color1"]
        color2 = self.target_criteria["color2"]
        
        # Find objects with specified colors
        obj1 = None
        obj2 = None
        for obj_data in self.objects.values():
            if obj_data["color"] == color1 and obj1 is None:
                obj1 = obj_data
            if obj_data["color"] == color2 and obj2 is None:
                obj2 = obj_data
        
        if "present" in self.task_instruction or "exists" in self.task_instruction:
            if obj1 is not None:
                return True, obj1
            elif obj2 is not None:
                return False, obj2
            else:
                return None
        
        if obj1 is None or obj2 is None:
            return None
            
        if "smaller" in self.task_instruction:
            condition_met = obj1["size_value"] < obj2["size_value"]
            target_dict = obj1 if condition_met else obj2
            return condition_met, target_dict
        elif "larger" in self.task_instruction:
            condition_met = obj1["size_value"] > obj2["size_value"]
            target_dict = obj1 if condition_met else obj2
            return condition_met, target_dict
            
        return None

    def _get_target_object(self):
        """Get target object based on current task phase"""
        if self.target_criteria["type"] == "conditional":
            condition_result = self._evaluate_condition()
            if condition_result:
                _, target_dict = condition_result
                return target_dict["actor"]
            return list(self.objects.values())[0]["actor"]
                
        elif self.target_criteria["type"] == "sequential":
            phase = self.execution_state.get("phase", "initial")
            if phase == "initial" or phase == "pick_first":
                color1 = self.target_criteria["color1"]
                for obj_data in self.objects.values():
                    if obj_data["color"] == color1:
                        return obj_data["actor"]
            elif phase == "pick_second":
                color2 = self.target_criteria["color2"]
                for obj_data in self.objects.values():
                    if obj_data["color"] == color2:
                        return obj_data["actor"]
            return list(self.objects.values())[0]["actor"]
            
        elif self.target_criteria["type"] == "reference":
            ref_type = self.target_criteria.get("reference_type")

            if ref_type == "same_as_largest":
                # Find the largest object
                largest_obj = max(self.objects.values(), key=lambda x: x["size_value"])
                largest_color = largest_obj["color"]
                
                # Find other objects with the same color that are not the largest object itself
                same_color_objs = [
                    obj for obj in self.objects.values() 
                    if obj["color"] == largest_color and obj["index"] != largest_obj["index"]
                ]
                
                if same_color_objs:
                    # If there are multiple objects with the same color, choose the largest one (besides the reference object)
                    return max(same_color_objs, key=lambda x: x["size_value"])["actor"]
                
                # If there are no other objects with the same color, return the largest object itself
                return largest_obj["actor"]

            elif ref_type == "same_as_smallest":
                # Find the smallest object
                smallest_obj = min(self.objects.values(), key=lambda x: x["size_value"])
                smallest_color = smallest_obj["color"]
                
                # Find other objects with the same color that are not the smallest object itself
                same_color_objs = [
                    obj for obj in self.objects.values()
                    if obj["color"] == smallest_color and obj["index"] != smallest_obj["index"]
                ]
                
                if same_color_objs:
                    # If there are multiple objects with the same color, choose the smallest one (besides the reference object)
                    return min(same_color_objs, key=lambda x: x["size_value"])["actor"]
                
                # If there are no other objects with the same color, return the smallest object itself
                return smallest_obj["actor"]

            elif ref_type == "size_match":
                ref_color = self.target_criteria.get("reference_color")
                ref_obj = None
                
                # Find the object with the reference color
                for obj_data in self.objects.values():
                    if obj_data["color"] == ref_color:
                        ref_obj = obj_data
                        break
                        
                if ref_obj:
                    ref_size = ref_obj["size_value"]
                    
                    # Find other objects that are closest in size to the reference object (excluding the reference object itself)
                    candidates = [
                        obj for obj in self.objects.values() 
                        if obj["index"] != ref_obj["index"]
                    ]
                    
                    if candidates:
                        # Choose the object with the closest size
                        target = min(candidates, key=lambda o: abs(o["size_value"] - ref_size))
                        return target["actor"]
                        
                return list(self.objects.values())[0]["actor"]

            elif ref_type == "smallest_not_color":
                excluded_color = self.target_criteria.get("excluded_color")
                
                # Find all objects that are not the specified color
                valid_objs = [
                    obj for obj in self.objects.values() 
                    if obj["color"] != excluded_color
                ]
                
                if valid_objs:
                    # Return the smallest object among them
                    smallest_valid = min(valid_objs, key=lambda x: x["size_value"])
                    return smallest_valid["actor"]
                    
                return list(self.objects.values())[0]["actor"]

        return list(self.objects.values())[0]["actor"]
        
    def _get_obs_extra(self, info: Dict):
        """Return extra observations with numeric types only"""
        # Convert execution phase to numeric
        phase_to_int = {
            "initial": 0,
            "pick_first": 1,
            "place_next": 2,
            "pick_second": 3,
            "completed": 4
        }
        current_phase = self.execution_state.get("phase", "initial")
        phase_int = phase_to_int.get(current_phase, 0)
        
        obs = dict(
            tcp_pose=self.agent.tcp_pose.raw_pose,
            execution_phase=torch.tensor([phase_int], dtype=torch.float32).repeat(self.num_envs, 1)
        )
        
        for obj_name, obj_data in self.objects.items():
            obs[f"{obj_name}_pose"] = obj_data["actor"].pose.raw_pose
            obs[f"is_grasping_{obj_name}"] = self.agent.is_grasping(obj_data["actor"])
        
        return obs

    def evaluate(self):
        """Evaluate based on task type"""
        device = self.device
        batch_size = self.num_envs
        
        if self.target_criteria["type"] == "conditional":
            # Check if condition was correctly evaluated and action taken
            condition_result = self._evaluate_condition()
            if condition_result is None:
                return {"success": torch.zeros(batch_size, dtype=torch.bool, device=device)}
                
            condition_met, target_obj = condition_result
            
            # Check if correct object is picked
            is_lifted = target_obj["actor"].pose.p[:, 2] > target_obj["size_value"] + self.lift_thresh
            is_grasped = self.agent.is_grasping(target_obj["actor"])
            return {"success": is_lifted & is_grasped}
                
        elif self.target_criteria["type"] == "sequential":
            # Check based on the specific command template
            color1 = self.target_criteria["color1"]
            color2 = self.target_criteria["color2"]
            
            obj1 = None
            obj2 = None
            for obj_data in self.objects.values():
                if obj_data["color"] == color1:
                    obj1 = obj_data
                if obj_data["color"] == color2:
                    obj2 = obj_data
                    
            if obj1 is None or obj2 is None:
                return {"success": torch.zeros(batch_size, dtype=torch.bool, device=device)}
            
            # Parse command to determine success criteria
            command_lower = self.task_instruction.lower()
                
            if "place it beside" in command_lower or "place it next to" in command_lower:
                # Check horizontal distance between objects
                dist = torch.linalg.norm(
                    obj1["actor"].pose.p[:, :2] - obj2["actor"].pose.p[:, :2], 
                    axis=1
                )
                
                # Calculate minimum distance to avoid collision (diagonal of cubes)
                obj1_diagonal = obj1["size_value"] * 1.414
                obj2_diagonal = obj2["size_value"] * 1.414
                min_dist = (obj1_diagonal + obj2_diagonal) / 2 + 0.002  # Small margin
                
                # Maximum distance to be considered "beside"  
                max_dist = 0.25  
                
                objects_adjacent = (dist >= min_dist) & (dist <= max_dist)
                
                # Check if objects are on table (with tolerance)
                table_height_tolerance = 0.04
                obj1_on_table = obj1["actor"].pose.p[:, 2] <= obj1["size_value"] + table_height_tolerance
                obj2_on_table = obj2["actor"].pose.p[:, 2] <= obj2["size_value"] + table_height_tolerance
                
                # Gripper should be empty
                gripper_open = ~self.agent.is_grasping(obj1["actor"]) & ~self.agent.is_grasping(obj2["actor"])
                
                return {"success": objects_adjacent & obj1_on_table & obj2_on_table & gripper_open}
            
            else:
                # Default evaluation for unrecognized sequential commands
                # Check if obj1 is near obj2
                dist = torch.linalg.norm(
                    obj1["actor"].pose.p[:, :2] - obj2["actor"].pose.p[:, :2], 
                    axis=1
                )
                next_to = dist < 0.10
                return {"success": next_to}
                
        elif self.target_criteria["type"] == "reference":
            ref_type = self.target_criteria.get("reference_type")
            target_obj = None

            if ref_type == "same_as_largest":
                largest_obj = max(self.objects.values(), key=lambda x: x["size_value"])
                largest_color = largest_obj["color"]
                for obj_data in self.objects.values():
                    if obj_data["color"] == largest_color and obj_data["index"] != largest_obj["index"]:
                        target_obj = obj_data["actor"]
                        break
                if target_obj is None:
                    target_obj = largest_obj["actor"]  # fallback

            elif ref_type == "same_as_smallest":
                smallest_obj = min(self.objects.values(), key=lambda x: x["size_value"])
                smallest_color = smallest_obj["color"]
                for obj_data in self.objects.values():
                    if obj_data["color"] == smallest_color and obj_data["index"] != smallest_obj["index"]:
                        target_obj = obj_data["actor"]
                        break
                if target_obj is None:
                    target_obj = smallest_obj["actor"]  # fallback

            elif ref_type == "size_match":
                ref_color = self.target_criteria.get("reference_color")
                ref_obj = None
                for obj_data in self.objects.values():
                    if obj_data["color"] == ref_color:
                        ref_obj = obj_data
                        break
                if ref_obj:
                    ref_size = ref_obj["size_value"]
                    candidates = [o for o in self.objects.values() if o["index"] != ref_obj["index"]]
                    if candidates:
                        target = min(candidates, key=lambda o: abs(o["size_value"] - ref_size))
                        target_obj = target["actor"]

            elif ref_type == "smallest_not_color":
                excluded_color = self.target_criteria.get("excluded_color")
                valid_objs = [obj for obj in self.objects.values() if obj["color"] != excluded_color]
                if valid_objs:
                    smallest_valid = min(valid_objs, key=lambda x: x["size_value"])
                    target_obj = smallest_valid["actor"]

            if target_obj is None:
                target_obj = list(self.objects.values())[0]["actor"]

            is_lifted = target_obj.pose.p[:, 2] > 0.03 + self.lift_thresh
            is_grasped = self.agent.is_grasping(target_obj)
            return {"success": is_lifted & is_grasped}
                    
        return {"success": torch.zeros(batch_size, dtype=torch.bool, device=device)}

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        """Compute reward based on task progress"""
        reward = torch.zeros_like(self.agent.tcp_pose.p[:, 0])
        
        if self.target_criteria["type"] == "conditional":
            condition_result = self._evaluate_condition()
            if condition_result:
                _, target_obj = condition_result
                # Reward for approaching correct object
                tcp_to_target = torch.linalg.norm(
                    target_obj["actor"].pose.p - self.agent.tcp_pose.p, axis=1
                )
                reward += 1 - torch.tanh(5 * tcp_to_target)
                
                # Reward for grasping
                if self.agent.is_grasping(target_obj["actor"]):
                    reward += 2
                        
        elif self.target_criteria["type"] == "sequential":
            # Reward based on current phase
            phase = self.execution_state.get("phase", "initial")
            if phase in ["initial", "pick_first"]:
                color1 = self.target_criteria["color1"]
                for obj_data in self.objects.values():
                    if obj_data["color"] == color1:
                        tcp_dist = torch.linalg.norm(
                            obj_data["actor"].pose.p - self.agent.tcp_pose.p, axis=1
                        )
                        reward += 1 - torch.tanh(5 * tcp_dist)
                        if self.agent.is_grasping(obj_data["actor"]):
                            reward += 3
                            self.execution_state["phase"] = "place_next"
                            
            elif phase == "place_next":
                color2 = self.target_criteria["color2"]
                for obj_data in self.objects.values():
                    if obj_data["color"] == color2:
                        # Reward for moving first object near second
                        if len(self.execution_state.get("picked_objects", [])) > 0:
                            picked_obj = self.execution_state["picked_objects"][0]
                            dist = torch.linalg.norm(
                                picked_obj.pose.p[:, :2] - obj_data["actor"].pose.p[:, :2], axis=1
                            )
                            reward += 2 - torch.tanh(3 * dist)
                            
        elif self.target_criteria["type"] == "reference":
            target_obj = self._get_target_object()
            # Reward for approaching and grasping correct object
            tcp_dist = torch.linalg.norm(target_obj.pose.p - self.agent.tcp_pose.p, axis=1)
            reward += 1 - torch.tanh(5 * tcp_dist)
            
            if self.agent.is_grasping(target_obj):
                reward += 3
                
        # Success bonus
        if info.get("success", False).any():
            reward += 5
            
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 12
    
    def get_task_instruction(self):
        return self.task_instruction


@register_env("Language-ConditionalPick-Hard", max_episode_steps=150)
class LanguageHardConditionalPickEnv(LanguageHardPickCubeBaseEnv):
    """Conditional logic: If the green cube is smaller than the red one, pick it up"""
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        self.current_task_type = "conditional"
        self._generate_task_instruction()

    def _load_scene(self, options: dict):
        if not hasattr(self, 'table_scene'):
            self.table_scene = TableSceneBuilder(self, robot_init_qpos_noise=self.robot_init_qpos_noise)
            self.table_scene.build()
        
        if not hasattr(self, 'objects'):
            self.objects = {}
            
            base_sizes = {
                "small": 0.010,
                "medium": 0.018,
                "large": 0.026
            }
            
            all_colors = list(self.object_colors.keys())
            selected_colors = random.sample(all_colors, random.randint(6, 7))
            
            object_configs = []
            
            compare_color1 = selected_colors[0]
            compare_color2 = selected_colors[1]
            
            if random.random() > 0.5:
                object_configs.append({
                    "color": compare_color1,
                    "size": "large",
                    "size_value": base_sizes["large"] + 0.002,
                    "role": "compare1"
                })
                object_configs.append({
                    "color": compare_color2,
                    "size": "small",
                    "size_value": base_sizes["small"],
                    "role": "compare2"
                })
            else:
                object_configs.append({
                    "color": compare_color1,
                    "size": "small",
                    "size_value": base_sizes["small"],
                    "role": "compare1"
                })
                object_configs.append({
                    "color": compare_color2,
                    "size": "large",
                    "size_value": base_sizes["large"] + 0.002,
                    "role": "compare2"
                })
            
            largest_color = selected_colors[2]
            object_configs.append({
                "color": largest_color,
                "size": "large",
                "size_value": base_sizes["large"] + 0.004,
                "role": "largest"
            })
            object_configs.append({
                "color": largest_color,
                "size": "medium",
                "size_value": base_sizes["medium"],
                "role": "largest_pair"
            })
            
            smallest_color = selected_colors[3]
            object_configs.append({
                "color": smallest_color,
                "size": "small",
                "size_value": base_sizes["small"] - 0.003, 
                "role": "smallest"
            })
            object_configs.append({
                "color": smallest_color,
                "size": "medium",
                "size_value": base_sizes["medium"] + 0.001,
                "role": "smallest_pair"
            })
            
            size_match_color1 = selected_colors[4]
            size_match_color2 = selected_colors[5]
            shared_size = base_sizes["small"] + 0.002
            
            object_configs.append({
                "color": size_match_color1,
                "size": "small",
                "size_value": shared_size,
                "role": "size_match1"
            })
            object_configs.append({
                "color": size_match_color2,
                "size": "small",
                "size_value": shared_size,
                "role": "size_match2"
            })
            
            random.shuffle(object_configs)
            
            for i, config in enumerate(object_configs):
                obj_name = f"cube{i}"
                
                builder = self.scene.create_actor_builder()
                builder.add_box_collision(half_size=[config["size_value"]] * 3)
                
                material = sapien.render.RenderMaterial()
                material.base_color = self.object_colors[config["color"]]
                builder.add_box_visual(half_size=[config["size_value"]] * 3, material=material)
                
                builder.initial_pose = sapien.Pose(p=[0, 0, 1.0])
                actor = builder.build(name=obj_name)
                
                self.objects[obj_name] = {
                    "actor": actor,
                    "color": config["color"],
                    "size": config["size"],
                    "size_value": config["size_value"],
                    "index": i,
                    "role": config.get("role", "generic")
                }

@register_env("Language-SequentialPlace-Hard", max_episode_steps=150)
class LanguageHardSequentialPlaceEnv(LanguageHardPickCubeBaseEnv):
    """Sequential task: First pick the red cube, then place it next to the blue one"""
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        self.current_task_type = "sequential"
        self._generate_task_instruction()

    def _load_scene(self, options: dict):
        if not hasattr(self, 'table_scene'):
            self.table_scene = TableSceneBuilder(self, robot_init_qpos_noise=self.robot_init_qpos_noise)
            self.table_scene.build()
        
        if not hasattr(self, 'objects'):
            self.objects = {}
            
            base_sizes = {
                "small": 0.010,
                "medium": 0.018,
                "large": 0.026
            }
            
            # Select colors for objects - each color appears only once
            all_colors = list(self.object_colors.keys())
            num_objects = 6  # Number of objects to create
            selected_colors = random.sample(all_colors, num_objects)
            
            # Create size variations for diversity
            size_variations = [
                ("small", base_sizes["small"] - 0.003),
                ("small", base_sizes["small"]),
                ("medium", base_sizes["medium"] - 0.002),
                ("medium", base_sizes["medium"] + 0.002),
                ("large", base_sizes["large"] - 0.001),
                ("large", base_sizes["large"] + 0.004),
            ]
            
            # Shuffle size variations for random assignment
            random.shuffle(size_variations)
            
            object_configs = []
            for i, color in enumerate(selected_colors):
                size_name, size_value = size_variations[i % len(size_variations)]
                object_configs.append({
                    "color": color,
                    "size": size_name,
                    "size_value": size_value
                })
            
            # Shuffle object configurations for random placement
            random.shuffle(object_configs)
            
            # Create objects
            for i, config in enumerate(object_configs):
                obj_name = f"cube{i}"
                
                builder = self.scene.create_actor_builder()
                builder.add_box_collision(half_size=[config["size_value"]] * 3)
                
                material = sapien.render.RenderMaterial()
                material.base_color = self.object_colors[config["color"]]
                builder.add_box_visual(half_size=[config["size_value"]] * 3, material=material)
                
                builder.initial_pose = sapien.Pose(p=[0, 0, 1.0])
                actor = builder.build(name=obj_name)
                
                self.objects[obj_name] = {
                    "actor": actor,
                    "color": config["color"],
                    "size": config["size"],
                    "size_value": config["size_value"],
                    "index": i
                }

@register_env("Language-ReferenceSelect-Hard", max_episode_steps=150)
class LanguageHardReferenceSelectEnv(LanguageHardPickCubeBaseEnv):
    """Reference-based: Pick based on size/color relationships"""  
    
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            
            # Reset execution state
            self.execution_state = {"phase": "initial", "picked_objects": [], "target_sequence": []}
            
            self.current_task_type = "reference"
            self.target_criteria = {"type": "reference"}
            
            self._generate_task_instruction()
            
            # Generate safe positions for objects
            safe_positions = self._generate_safe_positions(num_objects=len(self.objects))
            
            # Place one object near corner for reference tasks (optional)
            corner_idx = random.randint(0, len(self.objects) - 1)
            corner_position = [0.15, 0.15]  # Near corner
            safe_positions[corner_idx] = corner_position
            self.reference_object_idx = corner_idx
            
            for i, (obj_name, obj_data) in enumerate(self.objects.items()):
                obj_xyz = torch.zeros((b, 3))
                obj_xyz[:, :2] = torch.tensor(safe_positions[i]) + (torch.rand((b, 2)) * 2 - 1) * 0.01
                obj_xyz[:, 2] = obj_data["size_value"]
                obj_data["actor"].set_pose(Pose.create_from_pq(obj_xyz))


LanguageHardPickCubeBaseEnv.__doc__ = LANGUAGE_HARD_DOC_STRING