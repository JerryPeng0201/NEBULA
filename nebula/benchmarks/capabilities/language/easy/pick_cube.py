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

LANGUAGE_EASY_DOC_STRING = """Task Description:
Test basic language grounding ability with three task types:
1. Color-based selection: "Pick up the red cube"
2. Size-based selection: "Place the smallest cube" 
3. Object type reference: "Grab the blue block"

Language Commands:
- Task Set #1: Color commands - "Pick the red cube"
- Task Set #2: Size commands - "Pick the smallest cube",
- Task Set #3: Combined commands - "Grab the blue block"

Randomizations:
- Object colors, sizes, and positions are randomized
- Different phrasings of the same command type
- Multiple distractor objects with varying attributes
- Task type is randomly selected each episode

Success Conditions:
- Pick up the correct object specified in the language command
- Object is lifted above minimum threshold
- Robot maintains stability after task completion
"""

#@register_env("LanguageEasy-PickCube", max_episode_steps=100)
class LanguageEasyPickCubeBaseEnv(BaseEnv):
    
    SUPPORTED_ROBOTS = ["panda", "fetch"]
    agent: Union[Panda, Fetch]
    
    lift_thresh = 0.05
    
    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        
        # Object colors and names
        self.object_colors = {
            "red": [1, 0, 0, 1],
            "blue": [0, 0, 1, 1], 
            "green": [0, 1, 0, 1],
            "yellow": [1, 1, 0, 1],
            "purple": [1, 0, 1, 1]
        }
        
        base_sizes = {"small": 0.010, "medium": 0.015, "large": 0.020}
        self.object_sizes = {}
        
        for size_name, base_size in base_sizes.items():
            random_factor = 1.0 + (random.random() - 0.5) * 0.35
            self.object_sizes[size_name] = base_size * random_factor
            
        # Task types
        self.task_types = ["color", "size", "combined"]
        
        # Language templates for each task type
        self.command_templates = {
            "color": [
                "Pick up the {color} cube",
            ],
            "size": [
                "Pick the {size} cube",
            ],
            "combined": [
                "Grab the {color} block",
            ]
        }
        
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
        pose = sapien_utils.look_at([0.6, 0.6, 0.6], [0.0, 0.0, 0.2])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

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
    
    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.table_scene.build()
        
        # Create 5 objects with different colors and sizes
        self.objects = {}
        color_names = list(self.object_colors.keys())
        size_names = list(self.object_sizes.keys())
        
        for i in range(5):
            color_name = color_names[i]
            size_name = size_names[i % 3]  # Cycle through sizes
            size_value = self.object_sizes[size_name]
            color_value = self.object_colors[color_name]
            
            obj_name = f"{color_name}_{size_name}_cube"
            obj = actors.build_cube(
                self.scene,
                half_size=size_value,
                color=color_value,
                name=obj_name,
                initial_pose=sapien.Pose(p=[0, 0, 1.0]) 
            )
            self.objects[obj_name] = {
                "actor": obj,
                "color": color_name,
                "size": size_name,
                "size_value": size_value
            }

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            
            # Randomly select task type
            self.current_task_type = random.choice(self.task_types)
            
            # Generate language command based on task type
            if self.current_task_type == "color":
                self.target_color = random.choice(list(self.object_colors.keys()))
                template = random.choice(self.command_templates["color"])
                self.task_instruction = template.format(color=self.target_color)
                self.target_criteria = {"type": "color", "value": self.target_color}
                
            elif self.current_task_type == "size":
                self.target_size = random.choice(list(self.object_sizes.keys()))
                template = random.choice(self.command_templates["size"])
                self.task_instruction = template.format(size=self.target_size)
                self.target_criteria = {"type": "size", "value": self.target_size}
                
            else:  # combined
                self.target_color = random.choice(list(self.object_colors.keys()))
                template = random.choice(self.command_templates["combined"])
                self.task_instruction = template.format(color=self.target_color)
                self.target_criteria = {"type": "color", "value": self.target_color}
            
            print(f"Task Type: {self.current_task_type}")
            print(f"Language Command: '{self.task_instruction}'")
            print(f"Target Criteria: {self.target_criteria}")
            
            # Position objects in safe areas on the table
            safe_positions = [
                [0.25, 0.15],
                [0.25, -0.15],
                [0.05, 0.20],
                [0.05, -0.20],
                [0.15, 0.00]
            ]
            
            for i, (obj_name, obj_data) in enumerate(self.objects.items()):
                obj_xyz = torch.zeros((b, 3))
                if i < len(safe_positions):
                    obj_xyz[:, :2] = torch.tensor(safe_positions[i]) + (torch.rand((b, 2)) * 2 - 1) * 0.03
                else:
                    obj_xyz[:, :2] = torch.tensor([0.18, 0.0]) + (torch.rand((b, 2)) * 2 - 1) * 0.08
                obj_xyz[:, 2] = obj_data["size_value"]
                obj_data["actor"].set_pose(Pose.create_from_pq(obj_xyz))
            
            # Let physics settle
            for _ in range(10):
                self.scene.step()

    def _get_target_object(self):
        """Get the target object based on current task criteria"""
        if self.target_criteria["type"] == "color":
            target_color = self.target_criteria["value"]
            for obj_name, obj_data in self.objects.items():
                if obj_data["color"] == target_color:
                    return obj_data["actor"]
        elif self.target_criteria["type"] == "size":
            target_size = self.target_criteria["value"]
            if target_size == "small":
                # Find smallest object
                min_size = min(obj_data["size_value"] for obj_data in self.objects.values())
                for obj_data in self.objects.values():
                    if obj_data["size_value"] == min_size:
                        return obj_data["actor"]
            elif target_size == "large":
                # Find largest object
                max_size = max(obj_data["size_value"] for obj_data in self.objects.values())
                for obj_data in self.objects.values():
                    if obj_data["size_value"] == max_size:
                        return obj_data["actor"]
            else:  # medium
                # Find medium object
                sizes = sorted(set(obj_data["size_value"] for obj_data in self.objects.values()))
                if len(sizes) >= 3:
                    medium_size = sizes[1]  # Middle size
                    for obj_data in self.objects.values():
                        if obj_data["size_value"] == medium_size:
                            return obj_data["actor"]
        
        # Fallback: return first object
        return list(self.objects.values())[0]["actor"]

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            tcp_pose=self.agent.tcp_pose.raw_pose,
        )
        
        # Add all object poses and their attributes
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
        
        # 1. Reaching reward
        tcp_to_target_dist = torch.linalg.norm(
            target_obj.pose.p - self.agent.tcp_pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_target_dist)
        
        # 2. Grasping reward (higher for correct object)
        grasping_reward = self.agent.is_grasping(target_obj).float() * 2
        
        # 3. Lifting reward
        target_size = max(obj_data["size_value"] for obj_data in self.objects.values())
        lift_height = target_obj.pose.p[:, 2] - target_size
        lifting_reward = torch.clamp(lift_height / self.lift_thresh, 0, 1) * 2
        
        # 4. Language grounding penalty for wrong objects
        wrong_object_penalty = self._compute_wrong_object_penalty(target_obj)
        
        reward = reaching_reward + grasping_reward + lifting_reward - wrong_object_penalty
        
        # Success bonus
        reward[info["success"]] += 3
            
        return reward

    def _compute_wrong_object_penalty(self, target_obj):
        """Penalty for grasping objects not specified in language command"""
        penalty = torch.zeros_like(self.agent.tcp_pose.p[:, 0])
        for obj_data in self.objects.values():
            if obj_data["actor"] != target_obj:
                wrong_grasp = self.agent.is_grasping(obj_data["actor"]).float()
                penalty += wrong_grasp * 1.0  # Strong penalty for wrong object
        return penalty

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 8
    
    def get_task_instruction(self):
        return self.task_instruction

# Add docstrings to the classes
LanguageEasyPickCubeBaseEnv.__doc__ = LANGUAGE_EASY_DOC_STRING

@register_env("Language-PickCube-Easy", max_episode_steps=100)
class LanguageEasyPickCubeColorEnv(LanguageEasyPickCubeBaseEnv):
    """Fixed color-based task: Pick the red/blue/green cube"""
    
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            
            # Force color task
            self.current_task_type = "color"
            self.target_color = random.choice(["red", "blue", "green", "yellow", "purple"])
            
            # Generate color-based command
            template = random.choice(self.command_templates["color"])
            self.task_instruction = template.format(color=self.target_color)
            self.target_criteria = {"type": "color", "value": self.target_color}
            
            safe_positions = self._generate_safe_positions(num_objects=5)

            for i, (obj_name, obj_data) in enumerate(self.objects.items()):
                obj_xyz = torch.zeros((b, 3))
                obj_xyz[:, :2] = torch.tensor(safe_positions[i]) + (torch.rand((b, 2)) * 2 - 1) * 0.02
                obj_xyz[:, 2] = obj_data["size_value"]
                obj_data["actor"].set_pose(Pose.create_from_pq(obj_xyz))
                        

@register_env("Language-PlaceCube-Easy", max_episode_steps=100)
class LanguageEasyPickCubeSizeEnv(LanguageEasyPickCubeBaseEnv):
    """Fixed size-based task: Pick the smallest/largest cube"""

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.table_scene.build()
        
        # Create objects with guaranteed unique sizes
        self.objects = {}
        color_names = list(self.object_colors.keys())
        
        # Define 5 distinct sizes to ensure uniqueness
        distinct_sizes = [0.008, 0.012, 0.016, 0.020, 0.024]
        
        for i in range(5):
            color_name = color_names[i]
            size_value = distinct_sizes[i]
            color_value = self.object_colors[color_name]
            
            # Map size to category for target selection
            if size_value <= 0.010:
                size_category = "small"
            elif size_value >= 0.020:
                size_category = "large"
            else:
                size_category = "medium"
            
            obj_name = f"{color_name}_{size_category}_cube"
            obj = actors.build_cube(
                self.scene,
                half_size=size_value,
                color=color_value,
                name=obj_name,
                initial_pose=sapien.Pose(p=[0, 0, 1.0])
            )
            self.objects[obj_name] = {
                "actor": obj,
                "color": color_name,
                "size": size_category,
                "size_value": size_value
            }
    
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            
            # Force size task
            self.current_task_type = "size"
            self.target_size = random.choice(["small", "large"])  # Only small/large for clarity
            
            # Generate size-based command
            template = random.choice(self.command_templates["size"])
            # Map "small" to "smallest" and "large" to "largest" for natural language
            size_word = "smallest" if self.target_size == "small" else "largest"
            self.task_instruction = template.format(size=size_word)
            self.target_criteria = {"type": "size", "value": self.target_size}
            
            safe_positions = self._generate_safe_positions(num_objects=5,)

            for i, (obj_name, obj_data) in enumerate(self.objects.items()):
                obj_xyz = torch.zeros((b, 3))
                obj_xyz[:, :2] = torch.tensor(safe_positions[i]) + (torch.rand((b, 2)) * 2 - 1) * 0.02
                obj_xyz[:, 2] = obj_data["size_value"]
                obj_data["actor"].set_pose(Pose.create_from_pq(obj_xyz))


@register_env("Language-GrabBlock-Easy", max_episode_steps=100)
class LanguageEasyPickCubeCombinedEnv(LanguageEasyPickCubeBaseEnv):
    """Fixed combined task: Grab the blue block (color + object type)"""
    
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            
            # Force combined task
            self.current_task_type = "combined"
            self.target_color = random.choice(["red", "blue", "green"])
            
            # Generate combined command (color + object type)
            template = random.choice(self.command_templates["combined"])
            self.task_instruction = template.format(color=self.target_color)
            self.target_criteria = {"type": "color", "value": self.target_color}  # Same as color for selection
            
            safe_positions = self._generate_safe_positions(num_objects=5,)

            for i, (obj_name, obj_data) in enumerate(self.objects.items()):
                obj_xyz = torch.zeros((b, 3))
                obj_xyz[:, :2] = torch.tensor(safe_positions[i]) + (torch.rand((b, 2)) * 2 - 1) * 0.02
                obj_xyz[:, 2] = obj_data["size_value"]
                obj_data["actor"].set_pose(Pose.create_from_pq(obj_xyz))