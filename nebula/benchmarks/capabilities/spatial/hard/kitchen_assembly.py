from typing import Any, Dict, Union, List
import numpy as np
import sapien
import torch
import random
import os
import nebula.core.simulation.utils.randomization as randomization
from nebula.core.embodiment.robots import Fetch, Panda
from nebula.core.simulation.engine import BaseEnv
from nebula.core.sensors.camera import CameraConfig
from nebula.utils import sapien_utils
from nebula.utils.building import actors
from nebula.utils.registration import register_env
from nebula.utils.scene_builder.table import TableSceneBuilder
from nebula.utils.structs.pose import Pose

@register_env("Spatial-KitchenAssembly-Hard", max_episode_steps=150)
class SpatialHardKitchenAssemblyEnv(BaseEnv):
    """Task Description:
    Perform complex 3D spatial assembly tasks in a kitchen environment.
    The robot must manipulate kitchen objects (containers, utensils, and food) placed randomly on a table
    and follow spatial instructions to arrange them properly.

    Task Components:
    - Multiple YCB kitchen objects randomly placed on table
    - Simple multi-step spatial instructions
    - Basic spatial relationships (inside, beside, on shelf)

    Success Conditions:
    - Follow spatial instruction correctly
    - Stable object placement
    """

    SUPPORTED_ROBOTS = ["panda", "fetch"]
    agent: Union[Panda, Fetch]

    # Scene dimensions
    COUNTER_HEIGHT = 0.05
    UPPER_SHELF_HEIGHT = 0.35
    LOWER_SHELF_HEIGHT = 0.03

    # Tolerances
    PLACEMENT_TOLERANCE = 0.05
    STACKING_TOLERANCE = 0.03
    GROUPING_DISTANCE = 0.08

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        
        # YCB kitchen objects
        self.kitchen_containers = ["024_bowl", "025_mug", "029_plate"]
        self.kitchen_utensils = ["030_fork", "031_spoon", "032_knife"]
        self.kitchen_food = ["011_banana", "012_strawberry", "013_apple"]
        
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
        self.table_scene = TableSceneBuilder(self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.table_scene.build()
        
        # Ensure YCB assets are available
        from nebula import ASSET_DIR
        if not os.path.exists(ASSET_DIR / "assets/mani_skill2_ycb"):
            from nebula.utils import assets, download_asset
            download_asset.download(assets.DATA_SOURCES["ycb"])
        
        # Initialize containers
        self.selected_objects = {}
        self.kitchen_objects = {}

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # Clean up previous objects
            if hasattr(self, 'kitchen_objects'):
                for obj in self.kitchen_objects.values():
                    if obj is not None:
                        try:
                            self.scene.remove_actor(obj)
                        except:
                            pass
                self.kitchen_objects.clear()

            # Select and create objects
            self._select_kitchen_objects()
            self._create_kitchen_objects()
            self._build_kitchen_surfaces()

            # Generate instruction FIRST
            self._generate_kitchen_instruction()

            # THEN position objects to ensure instruction is not initially satisfied
            self._position_objects_randomly(b)

            # Calculate targets after positioning
            self._calculate_target_positions()

            self.kitchen_areas = {
                "counter": {"height": self.COUNTER_HEIGHT, "area": [-0.35, 0.35, -0.35, 0.35]}
            }
            
            """print(f"Kitchen Instruction: '{self.task_instruction}'")
            print(f"Selected Objects: {list(self.selected_objects.keys())}")"""

            # Brief final settling
            for _ in range(5):
                self.scene.step()

    def _select_kitchen_objects(self):
        """Select 3-4 kitchen objects"""
        self.selected_objects = {
            "container": random.choice(self.kitchen_containers),
            "utensil": random.choice(self.kitchen_utensils),
            "food": random.choice(self.kitchen_food)
        }
        
        # Sometimes add second container
        if random.random() > 0.4:
            available_containers = [c for c in self.kitchen_containers if c != self.selected_objects["container"]]
            self.selected_objects["container2"] = random.choice(available_containers)

    def _create_kitchen_objects(self):
        """Create YCB objects with enhanced stability"""
        self.kitchen_objects = {}
        episode_id = random.randint(10000, 99999)
        
        for obj_type, ycb_id in self.selected_objects.items():
            builder = actors.get_actor_builder(self.scene, id=f"ycb:{ycb_id}")
            builder.initial_pose = sapien.Pose(p=[0, 0, 1.0])
            
            unique_name = f"kitchen_{obj_type}_{episode_id}_{random.randint(1000, 9999)}"
            obj = builder.build(name=unique_name)
            
            # Set physics properties
            if obj.has_collision_shapes:
                for shape in obj._bodies[0].get_collision_shapes():
                    material = shape.get_physical_material()
                    material.static_friction = 0.8
                    material.dynamic_friction = 0.7
                    material.restitution = 0.1
                    
            self.kitchen_objects[obj_type] = obj

    def _build_kitchen_surfaces(self):
        surface_id = random.randint(10000, 99999)
        workspace_builder = self.scene.create_actor_builder()
        workspace_builder.add_box_visual(
            half_size=[0.35, 0.35, 0.001],
            material=sapien.render.RenderMaterial(
                base_color=[0.9, 0.9, 0.8, 0.8],
                metallic=0.1,
                roughness=0.7
            )
        )
        workspace_builder.set_physx_body_type("kinematic")
        workspace_builder.initial_pose = sapien.Pose(p=[0, 0, 0])
        self.workspace_area = workspace_builder.build(name=f"workspace_area_{surface_id}")

    def _position_objects_randomly(self, b):
        """Position objects far apart"""
        used_positions = []
        min_distance = 0.2
        
        for i, (obj_type, obj) in enumerate(self.kitchen_objects.items()):
            # Find random position far from existing objects
            for _ in range(50):
                x = random.uniform(-0.20, 0.20)
                y = random.uniform(-0.20, 0.20)
                pos = [x, y]
                
                if all(np.linalg.norm(np.array(pos) - np.array(used_pos)) > min_distance
                       for used_pos in used_positions):
                    used_positions.append(pos)
                    break
            else:
                fallback_positions = [
                    [-0.20, -0.20], [0.20, -0.20], [-0.20, 0.20], [0.20, 0.20], [0.0, -0.35]
                ]
                pos = fallback_positions[i % len(fallback_positions)]
                used_positions.append(pos)

            obj_xyz = torch.zeros((b, 3))
            obj_xyz[:, 0] = pos[0]
            obj_xyz[:, 1] = pos[1]
            obj_xyz[:, 2] = self.COUNTER_HEIGHT + 0.01
            
            quat = torch.zeros((b, 4))
            quat[:, 3] = 1.0
            
            obj.set_pose(Pose.create_from_pq(p=obj_xyz, q=quat))
            obj.set_linear_velocity(torch.zeros((b, 3)))
            obj.set_angular_velocity(torch.zeros((b, 3)))

            # Let settle
            for _ in range(10):
                self.scene.step()

    def _generate_kitchen_instruction(self):
        """Generate logical kitchen instructions"""
        container_type = self._get_container_type(self.selected_objects["container"])
        container2_type = self._get_container_type(self.selected_objects.get("container2", "")) if "container2" in self.selected_objects else None
        utensil_name = self.selected_objects["utensil"].split("_")[1]
        food_name = self.selected_objects["food"].split("_")[1]

        templates = []
        
        # Food in container + additional action
        if self._is_food_container_compatible(food_name, container_type):
            templates.append(f"Put the {food_name} in the {container_type} and place the {utensil_name} beside the {container_type}")
            
        if container2_type:
            templates.append(f"Put the {food_name} in the {container_type} and place the {container2_type} beside it")

        # Multiple beside relationships
        templates.append(f"Place the {utensil_name} beside the {container_type} and place the {food_name} beside the {utensil_name}")

        self.task_instruction = random.choice(templates) if templates else f"Place the {utensil_name} beside the {container_type}"
        self._parse_instruction_steps()

    def _get_container_type(self, ycb_id):
        """Get container type name"""
        if "bowl" in ycb_id:
            return "bowl"
        elif "plate" in ycb_id:
            return "plate"
        elif "mug" in ycb_id:
            return "mug"
        return "container"

    def _is_food_container_compatible(self, food_name, container_type):
        """Check food-container compatibility"""
        compatibility = {
            "bowl": ["banana", "strawberry", "apple"],
            "plate": ["banana", "strawberry", "apple"],
            "mug": []
        }
        return food_name in compatibility.get(container_type, [])

    def _parse_instruction_steps(self):
        """Parse instruction into actionable steps"""
        self.instruction_steps = []
        instruction_lower = self.task_instruction.lower()
        
        if "put" in instruction_lower and "in the" in instruction_lower:
            self.instruction_steps.append({"action": "place_inside", "objects": ["food", "container"]})
            
        if "beside" in instruction_lower:
            if "utensil" in instruction_lower and "container" in instruction_lower:
                self.instruction_steps.append({"action": "place_beside", "objects": ["utensil", "container"]})
            elif "utensil" in instruction_lower and "food" in instruction_lower:
                self.instruction_steps.append({"action": "place_beside", "objects": ["utensil", "food"]})
                
        if not self.instruction_steps:
            self.instruction_steps.append({"action": "place_beside", "objects": ["utensil", "container"]})

    def _calculate_target_positions(self):
        """Calculate target positions for evaluation"""
        self.target_positions = {}
        for step in self.instruction_steps:
            if step["action"] == "place_inside":
                self.target_positions["food_in_container"] = True
            elif step["action"] == "place_beside":
                self.target_positions["beside_relationship"] = step["objects"]

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )
        
        if "state" in self.obs_mode:
            for obj_type, obj in self.kitchen_objects.items():
                obs[f"{obj_type}_pose"] = obj.pose.raw_pose
                obs[f"is_grasping_{obj_type}"] = self.agent.is_grasping(obj)
                obs[f"tcp_to_{obj_type}_pos"] = obj.pose.p - self.agent.tcp.pose.p
                
        return obs

    def evaluate(self):
        """Evaluate task completion"""
        step_completions = []
        
        for step in self.instruction_steps:
            if step["action"] == "place_inside":
                completion = self._check_inside_relationship(step["objects"])
            elif step["action"] == "place_beside":
                completion = self._check_beside_relationship(step["objects"])
            else:
                completion = torch.tensor(False, device=self.device, dtype=torch.bool)
                
            step_completions.append(completion)

        all_stable = self._check_all_objects_stable()
        overall_success = torch.tensor(all(step_completions) and all_stable, device=self.device, dtype=torch.bool)

        return {
            "success": overall_success,
            "steps_completed": sum(step_completions),
            "total_steps": len(self.instruction_steps),
            "all_stable": all_stable,
            "task_instruction": self.task_instruction,
            "step_details": step_completions
        }

    def _check_inside_relationship(self, objects):
        """Check inside relationship"""
        if len(objects) < 2:
            return torch.tensor(False, device=self.device, dtype=torch.bool)
            
        food_obj = self.kitchen_objects.get(objects[0])
        container_obj = self.kitchen_objects.get(objects[1])
        
        if food_obj is None or container_obj is None:
            return torch.tensor(False, device=self.device, dtype=torch.bool)

        food_pos = food_obj.pose.p[0] if food_obj.pose.p.dim() > 1 else food_obj.pose.p
        container_pos = container_obj.pose.p[0] if container_obj.pose.p.dim() > 1 else container_obj.pose.p
        
        xy_distance = torch.linalg.norm(food_pos[:2] - container_pos[:2])
        height_in_container = (container_pos[2] + 0.01 < food_pos[2] < container_pos[2] + 0.1)
        
        result = (xy_distance < 0.03) and height_in_container
        return torch.tensor(result, device=self.device, dtype=torch.bool)

    def _check_beside_relationship(self, objects):
        """Check beside relationship"""
        if len(objects) < 2:
            return torch.tensor(False, device=self.device, dtype=torch.bool)
            
        obj1 = self.kitchen_objects.get(objects[0])
        obj2 = self.kitchen_objects.get(objects[1])
        
        if obj1 is None or obj2 is None:
            return torch.tensor(False, device=self.device, dtype=torch.bool)

        pos1 = obj1.pose.p[0] if obj1.pose.p.dim() > 1 else obj1.pose.p
        pos2 = obj2.pose.p[0] if obj2.pose.p.dim() > 1 else obj2.pose.p
        
        distance = torch.linalg.norm(pos1[:2] - pos2[:2])
        height_similar = abs(pos1[2] - pos2[2]) < self.PLACEMENT_TOLERANCE
        
        return torch.tensor((distance < self.GROUPING_DISTANCE) and height_similar,
                          device=self.device, dtype=torch.bool)

    def _check_all_objects_stable(self):
        """Check if all objects are stable"""
        for obj in self.kitchen_objects.values():
            linear_vel = obj.linear_velocity[0] if obj.linear_velocity.dim() > 1 else obj.linear_velocity
            angular_vel = obj.angular_velocity[0] if obj.angular_velocity.dim() > 1 else obj.angular_velocity
            
            linear_velocity = torch.linalg.norm(linear_vel)
            angular_velocity = torch.linalg.norm(angular_vel)
            
            if linear_velocity > 0.05 or angular_velocity > 0.1:
                return torch.tensor(False, device=self.device, dtype=torch.bool)
                
        return torch.tensor(True, device=self.device, dtype=torch.bool)

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        total_reward = 0.0
        
        # Grasping reward
        relevant_grasping = False
        for obj in self.kitchen_objects.values():
            if self.agent.is_grasping(obj):
                relevant_grasping = True
                break
        total_reward += 1.0 if relevant_grasping else 0.0

        # Progress reward
        progress_reward = self._calculate_progress_reward()
        total_reward += progress_reward

        # Step completion reward
        total_reward += info["steps_completed"] * 3.0

        # Stability reward
        if info["all_stable"]:
            total_reward += 2.0

        # Success reward
        if info["success"]:
            total_reward += 10.0

        return total_reward

    def _calculate_progress_reward(self):
        """Calculate progress towards completing steps"""
        progress = 0.0
        for step in self.instruction_steps:
            if step["action"] == "place_beside":
                obj1 = self.kitchen_objects.get(step["objects"][0])
                obj2 = self.kitchen_objects.get(step["objects"][1])
                if obj1 and obj2:
                    distance = torch.linalg.norm(obj1.pose.p[0][:2] - obj2.pose.p[0][:2])
                    progress += max(0, 1.0 - distance / 0.5)
        return progress

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        """Compute normalized dense reward"""
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 20.0
    
    def get_task_instruction(self):
        return self.task_instruction