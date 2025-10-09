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

LANGUAGE_MEDIUM_DOC_STRING = """Task Description:
Test advanced language grounding with relative position analysis and selective instruction comprehension:
1. Multi-attribute selection: "Pick the small green cube" 
2. Negative selection: "Place the cube that is not red"
3. Exclusion-based selection: "The red cube will not be used. Place another cube"

Language Commands:
- Task Set #1: Multi-attribute commands - "Pick the small green cube", "Grab the large blue object"
- Task Set #2: Negative selection commands - "Place the cube that is not red", "Take the object that is not yellow"
- Task Set #3: Exclusion commands - "The red cube will not be used. Place another cube", "Ignore the blue one. Pick a different cube"

Randomizations:
- Object colors, sizes, and positions are randomized
- Multiple objects with overlapping attributes as distractors
- Task type is randomly selected each episode
- Complex attribute combinations require precise understanding

Success Conditions:
- Pick up the correct object based on complex language criteria
- Object is lifted above minimum threshold
- Robot maintains stability after task completion
- Demonstrate understanding of negation and exclusion
"""

class LanguageMediumPickCubeBaseEnv(BaseEnv):
    
    SUPPORTED_ROBOTS = ["panda", "fetch"]
    agent: Union[Panda, Fetch]
    
    lift_thresh = 0.05
    
    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        
        # Object colors
        self.object_colors = {
            "red": [1, 0, 0, 1],
            "blue": [0, 0, 1, 1], 
            "green": [0, 1, 0, 1],
            "yellow": [1, 1, 0, 1],
            "purple": [1, 0, 1, 1]
        }
        
        # Object sizes with randomization
        base_sizes = {"small": 0.012, "medium": 0.017, "large": 0.022}
        self.object_sizes = {}
        for size_name, base_size in base_sizes.items():
            random_factor = 1.0 + (random.random() - 0.5) * 0.3
            self.object_sizes[size_name] = base_size * random_factor
            
        # Task types for medium difficulty
        self.task_types = ["multi_attribute", "negative_selection", "exclusion"]
        
        # Language templates for each task type
        self.command_templates = {
            "multi_attribute": [
                "Pick the {size} {color} cube",
                "Grab the {size} {color} object",
                "Take the {size} {color} block"
            ],
            "negative_selection": [
                "Place the cube that is not {color}",
                "Take the object that is not {color}",
                "Pick the cube that isn't {color}"
            ],
            "exclusion": [
                "The {color} cube will not be used. Place another cube",
                "Ignore the {color} one. Pick a different cube",
                "Don't use the {color} cube. Choose another one"
            ]
        }
        self.task_instruction = ""
        
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

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.table_scene.build()
        
        # Create 6 objects with varied attributes for more complex selection
        self.objects = {}
        color_names = list(self.object_colors.keys())
        size_names = list(self.object_sizes.keys())
        
        # Ensure we have varied combinations
        combinations = [
            ("red", "small"), ("blue", "large"), ("green", "small"), 
            ("yellow", "medium"), ("purple", "large"), ("red", "medium")
        ]
        
        for i, (color_name, size_name) in enumerate(combinations):
            size_value = self.object_sizes[size_name]
            color_value = self.object_colors[color_name]
            
            obj_name = f"{color_name}_{size_name}_cube"
            obj = actors.build_cube(
                self.scene, half_size=size_value, color=color_value, 
                name=obj_name, initial_pose=sapien.Pose(p=[0, 0, 1.0])
            )
            self.objects[obj_name] = {
                "actor": obj, "color": color_name, "size": size_name, "size_value": size_value
            }

    def _generate_safe_positions(
        self,
        num_objects: int,
        min_clearance: float = 0.10,   
        edge_margin: float = 0.03, 
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

        radii = []
        for _, obj_data in list(self.objects.items())[:num_objects]:
            r_obj = float(obj_data["size_value"]) + 0.003
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
                if d < (radii[j] + radii[idx] + min_clearance):
                    return False
            return True

        placed = []
        trials = 0
        step = 2.0 * float(radii.max()) + min_clearance
        nx = max(2, int((x_hi - x_lo) / step))
        ny = max(2, int((y_hi - y_lo) / step))

        while len(placed) < num_objects and trials < max_trials:
            trials += 1
            gx = rng.integers(0, nx + 1)
            gy = rng.integers(0, ny + 1)
            x = x_lo + (x_hi - x_lo) * (gx / max(1, nx))
            y = y_lo + (y_hi - y_lo) * (gy / max(1, ny))
            x += (rng.random() - 0.5) * (0.5 * step)
            y += (rng.random() - 0.5) * (0.5 * step)
            x = float(np.clip(x, x_lo, x_hi))
            y = float(np.clip(y, y_lo, y_hi))

            idx = len(placed)
            if in_bounds(x, y) and is_reachable(x, y) and far_enough((x, y), placed, idx):
                placed.append((x, y))

        if len(placed) == num_objects:
            return [list(p) for p in placed]

        nx = max(2, int((x_hi - x_lo) / (2.0 * float(radii.max()) + min_clearance)))
        ny = max(2, int((y_hi - y_lo) / (2.0 * float(radii.max()) + min_clearance)))
        xs = np.linspace(x_lo, x_hi, nx)
        ys = np.linspace(y_lo, y_hi, ny)
        fallback = []
        for yv in ys:
            for xv in xs:
                if is_reachable(float(xv), float(yv)):
                    fallback.append([float(xv), float(yv)])
        
        if len(fallback) < num_objects:
            for _ in range(num_objects - len(fallback)):
                x = rng.uniform(x_lo, x_hi)
                y = rng.uniform(y_lo, y_hi)
                fallback.append([float(x), float(y)])
        
        rng.shuffle(fallback)
        fallback = fallback[:num_objects]

        jitter = min(0.5 * min_clearance, 0.01)
        out = []
        for xv, yv in fallback:
            dx = (rng.random() - 0.5) * 2 * jitter
            dy = (rng.random() - 0.5) * 2 * jitter
            x = float(np.clip(xv + dx, x_lo, x_hi))
            y = float(np.clip(yv + dy, y_lo, y_hi))
            out.append([x, y])
        
        assert len(out) == num_objects, f"Expected {num_objects} positions, got {len(out)}"
        return out

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            
            # Randomly select task type
            self.current_task_type = random.choice(self.task_types)
            self._generate_task_instruction()
            
            # Generate safe positions
            safe_positions = self._generate_safe_positions(num_objects=len(self.objects))
            
            for i, (obj_name, obj_data) in enumerate(self.objects.items()):
                obj_xyz = torch.zeros((b, 3))
                obj_xyz[:, :2] = torch.tensor(safe_positions[i]) + (torch.rand((b, 2)) * 2 - 1) * 0.02
                obj_xyz[:, 2] = obj_data["size_value"]
                obj_data["actor"].set_pose(Pose.create_from_pq(obj_xyz))

    def _generate_task_instruction(self):
        """Generate language command based on task type and available objects"""
        if self.current_task_type == "multi_attribute":
            available_combinations = [(obj_data["color"], obj_data["size"]) 
                                    for obj_data in self.objects.values()]
            target_color, target_size = random.choice(available_combinations)
            
            template = random.choice(self.command_templates["multi_attribute"])
            self.task_instruction = template.format(color=target_color, size=target_size)
            self.target_criteria = {"type": "multi_attribute", "color": target_color, "size": target_size}
            
        elif self.current_task_type == "negative_selection":
            available_colors = set(obj_data["color"] for obj_data in self.objects.values())
            if len(available_colors) <= 1:
                self.current_task_type = "multi_attribute"
                self._generate_task_instruction()
                return
                
            excluded_color = random.choice(list(available_colors))
            while all(obj_data["color"] == excluded_color for obj_data in self.objects.values()):
                excluded_color = random.choice(list(available_colors))
            
            valid_objects = [obj_name for obj_name, obj_data in self.objects.items() 
                            if obj_data["color"] != excluded_color]
            target_object_name = random.choice(valid_objects)
            
            template = random.choice(self.command_templates["negative_selection"])
            self.task_instruction = template.format(color=excluded_color)
            self.target_criteria = {
                "type": "negative_selection", 
                "excluded_color": excluded_color,
                "target_object_name": target_object_name  
            }
            
        else: 
            available_colors = set(obj_data["color"] for obj_data in self.objects.values())
            if len(available_colors) <= 1:
                self.current_task_type = "multi_attribute"
                self._generate_task_instruction()
                return
                
            excluded_color = random.choice(list(available_colors))
            while all(obj_data["color"] == excluded_color for obj_data in self.objects.values()):
                excluded_color = random.choice(list(available_colors))

            valid_objects = [obj_name for obj_name, obj_data in self.objects.items() 
                            if obj_data["color"] != excluded_color]
            target_object_name = random.choice(valid_objects)
            
            template = random.choice(self.command_templates["exclusion"])
            self.task_instruction = template.format(color=excluded_color)
            self.target_criteria = {
                "type": "exclusion", 
                "excluded_color": excluded_color,
                "target_object_name": target_object_name
            }

    def _get_target_object(self):
        """Get target object based on language criteria with better selection logic"""
        if self.target_criteria["type"] == "multi_attribute":
            target_color = self.target_criteria["color"]
            target_size = self.target_criteria["size"]
            for obj_data in self.objects.values():
                if obj_data["color"] == target_color and obj_data["size"] == target_size:
                    return obj_data["actor"]
            print(f"ERROR: No object matches {target_color} {target_size}")
            return list(self.objects.values())[0]["actor"]
                    
        elif self.target_criteria["type"] in ["negative_selection", "exclusion"]:
            if "target_object_name" in self.target_criteria:
                target_name = self.target_criteria["target_object_name"]
                return self.objects[target_name]["actor"]
            
            excluded_color = self.target_criteria["excluded_color"]
            for obj_data in self.objects.values():
                if obj_data["color"] != excluded_color:
                    return obj_data["actor"]
            
            print(f"ERROR: No objects available that are not {excluded_color}")
            return list(self.objects.values())[0]["actor"]

    def _get_obs_extra(self, info: Dict):
        obs = dict(tcp_pose=self.agent.tcp_pose.raw_pose)
        for obj_name, obj_data in self.objects.items():
            obs[f"{obj_name}_pose"] = obj_data["actor"].pose.raw_pose
            obs[f"is_grasping_{obj_name}"] = self.agent.is_grasping(obj_data["actor"])
        return obs

    def evaluate(self):
        target_obj = self._get_target_object()
        target_obj_size = self.object_sizes["small"]
        for obj_data in self.objects.values():
            if obj_data["actor"] == target_obj:
                target_obj_size = obj_data["size_value"]
                break
        
        is_lifted = target_obj.pose.p[:, 2] > target_obj_size + self.lift_thresh
        is_grasped = self.agent.is_grasping(target_obj)
        is_robot_static = self.agent.is_static(0.2)
        success = is_lifted & is_grasped & is_robot_static
        
        return {"success": success, "is_lifted": is_lifted, "is_grasped": is_grasped, "is_robot_static": is_robot_static}

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        target_obj = self._get_target_object()
        
        # Reaching reward
        tcp_to_target_dist = torch.linalg.norm(target_obj.pose.p - self.agent.tcp_pose.p, axis=1)
        reaching_reward = 1 - torch.tanh(5 * tcp_to_target_dist)
        
        # Grasping reward
        grasping_reward = self.agent.is_grasping(target_obj).float() * 2
        
        # Lifting reward
        target_size = max(obj_data["size_value"] for obj_data in self.objects.values())
        lift_height = target_obj.pose.p[:, 2] - target_size
        lifting_reward = torch.clamp(lift_height / self.lift_thresh, 0, 1) * 2
        
        # Wrong object penalty (higher for medium difficulty)
        wrong_object_penalty = self._compute_wrong_object_penalty(target_obj) * 1.5
        
        reward = reaching_reward + grasping_reward + lifting_reward - wrong_object_penalty
        reward[info["success"]] += 4  # Higher success bonus
        return reward

    def _compute_wrong_object_penalty(self, target_obj):
        """Enhanced penalty for wrong object selection in medium difficulty"""
        penalty = torch.zeros_like(self.agent.tcp_pose.p[:, 0])
        for obj_data in self.objects.values():
            if obj_data["actor"] != target_obj:
                wrong_grasp = self.agent.is_grasping(obj_data["actor"]).float()
                penalty += wrong_grasp * 1.2  # Higher penalty than easy
        return penalty

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 10
    
    def get_task_instruction(self):
        return self.task_instruction


@register_env("Language-PickCube-Medium", max_episode_steps=200)
class LanguageMediumPickCubeEnv(LanguageMediumPickCubeBaseEnv):
    """Multi-attribute selection: Pick the small green cube"""
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            
            self.current_task_type = "multi_attribute"
            self._generate_task_instruction()
            
            safe_positions = self._generate_safe_positions(num_objects=len(self.objects))
            for i, (obj_name, obj_data) in enumerate(self.objects.items()):
                obj_xyz = torch.zeros((b, 3))
                obj_xyz[:, :2] = torch.tensor(safe_positions[i]) + (torch.rand((b, 2)) * 2 - 1) * 0.01
                obj_xyz[:, 2] = obj_data["size_value"]
                obj_data["actor"].set_pose(Pose.create_from_pq(obj_xyz))

@register_env("Language-PlaceCube-Medium", max_episode_steps=200)
class LanguageMediumPlaceCubeEnv(LanguageMediumPickCubeBaseEnv):
    """Negative selection: Place the cube that is not red"""
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            
            self.current_task_type = "negative_selection"
            self._generate_task_instruction()
            
            safe_positions = self._generate_safe_positions(num_objects=len(self.objects))
            for i, (obj_name, obj_data) in enumerate(self.objects.items()):
                obj_xyz = torch.zeros((b, 3))
                obj_xyz[:, :2] = torch.tensor(safe_positions[i]) + (torch.rand((b, 2)) * 2 - 1) * 0.01
                obj_xyz[:, 2] = obj_data["size_value"]
                obj_data["actor"].set_pose(Pose.create_from_pq(obj_xyz))

@register_env("Language-GrabBlock-Medium", max_episode_steps=200)
class LanguageMediumGrabBlockEnv(LanguageMediumPickCubeBaseEnv):
    """Exclusion: The red cube will not be used. Place another cube"""
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            
            self.current_task_type = "exclusion"
            self._generate_task_instruction()
            
            safe_positions = self._generate_safe_positions(num_objects=len(self.objects))
            for i, (obj_name, obj_data) in enumerate(self.objects.items()):
                obj_xyz = torch.zeros((b, 3))
                obj_xyz[:, :2] = torch.tensor(safe_positions[i]) + (torch.rand((b, 2)) * 2 - 1) * 0.01
                obj_xyz[:, 2] = obj_data["size_value"]
                obj_data["actor"].set_pose(Pose.create_from_pq(obj_xyz))

LanguageMediumPickCubeBaseEnv.__doc__ = LANGUAGE_MEDIUM_DOC_STRING