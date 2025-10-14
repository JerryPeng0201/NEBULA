"""
nebula_dataset_integration.py

Modified dataset.py to support both RLDS and Nebula datasets.
Replace your existing dataset.py with this modified version.
"""

import os
import torch
import itertools
from pathlib import Path
from PIL import Image, ImageFile
from torch.utils.data import IterableDataset

IGNORE_INDEX = -100
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

from .oxe import OXE_NAMED_MIXTURES, get_oxe_dataset_kwargs_and_weights
from .utils.data_utils import NormalizationType, save_dataset_statistics
from .rlds import dataset_statistics, build_interleaved_dataset

# Import nebula components
from .nebula_dataset import make_nebula_dataset, nebula_dataset_transform

class NebulaIterableDataset(IterableDataset):
    """Custom dataset for Nebula HDF5 format."""

    def __init__(
        self,
        data_root_dir,
        output_dir,
        task_names,
        task_instruction_map,
        image_size=224,
        max_length=1024,
        is_train=True,
        shuffle_buffer_size=1000_000,
        obs_backward_steps=0,
        obs_backward_delta=1,
        action_forward_steps=0,
        max_trajectories_per_task=None,
        vla_processor=None,
    ):
        super(NebulaIterableDataset, self).__init__()
        self.data_root_dir = data_root_dir
        self.task_names = task_names
        self.task_instruction_map = task_instruction_map
        self.image_size = image_size
        self.max_length = max_length
        self.is_train = is_train
        self.max_trajectories_per_task = max_trajectories_per_task
        self.vla_processor = vla_processor
        self.use_raw_dataloader = False

        self.total_ranks = torch.distributed.get_world_size()
        self.current_rank = torch.distributed.get_rank()

        # Load nebula dataset
        print(f"Loading Nebula dataset from {data_root_dir}...")
        self.trajectories, self.ds_stats = make_nebula_dataset(
            data_root_dir=data_root_dir,
            task_names=task_names,
            task_instruction_map=task_instruction_map,
            train=is_train,
            shuffle_seed=3407 * self.current_rank,
            max_trajectories_per_task=max_trajectories_per_task,
        )

        # Save dataset statistics
        self.ds_stats_pc = save_dataset_statistics(
            {"nebula": self.ds_stats}, 
            Path(output_dir) / "nebula_ds_stats.json"
        )

        print(f"Loaded {len(self.trajectories)} trajectories from Nebula dataset")

    def __len__(self):
        return len(self.trajectories)

    def _process_trajectory(self, trajectory):
        """Process a single trajectory into model input format - frame by frame."""
        # Apply transform to match expected format
        trajectory = nebula_dataset_transform(trajectory)
        
        # Extract observations and actions
        obs = trajectory["observation"]
        actions = torch.from_numpy(trajectory["action"])  # [T, 7]
        
        # Get language instruction
        lang_bytes = trajectory["language_instruction"][0]
        if isinstance(lang_bytes, bytes):
            lang = lang_bytes.decode('utf-8')
        else:
            lang = str(lang_bytes)
        lang = lang.lower()

        # Process single frame (you can randomize which frame to use)
        frame_idx = 0  # Or use: np.random.randint(0, len(actions))
        
        # Extract single frame
        image_data = obs["image_primary"]
        img_bytes = image_data[frame_idx]
        
        if isinstance(img_bytes, bytes) and len(img_bytes) > 0:
            import cv2
            import numpy as np
            nparr = np.frombuffer(img_bytes, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            current_image = Image.fromarray(img_rgb)
        else:
            current_image = Image.new('RGB', (self.image_size, self.image_size), (0, 0, 0))

        # Single action for this frame
        single_action = actions[frame_idx:frame_idx+1]  # [1, 7]

        # Process with VLA processor - single image, single action
        if self.vla_processor is not None:
            ret = self.vla_processor(
                text=lang,
                images=[current_image],  # Single image
                suffix_actions=single_action,  # Single action
                return_tensors="pt",
                padding=False,
                max_length=self.max_length,
                truncation=True,
                do_normalize=False,
            )

            model_inputs = dict(
                input_ids=ret["input_ids"][0],
                labels=ret["labels"][0],
                token_type_ids=ret["token_type_ids"][0],
                attention_mask=ret["attention_mask"][0],
                pixel_values=ret["pixel_values"],
                intrinsic=ret.get("intrinsic", torch.eye(3)),
                actions=single_action[0],  # [7]
            )
        else:
            model_inputs = dict(
                images=[current_image],
                text=lang,
                actions=single_action[0],
            )

        return model_inputs

    def __iter__(self):
        # Shuffle trajectories for training
        if self.is_train:
            import random
            random.seed(3407 * self.current_rank)
            trajectories = random.sample(self.trajectories, len(self.trajectories))
        else:
            trajectories = self.trajectories

        # Distribute across workers
        if torch.utils.data.get_worker_info() is not None:
            worker_total_num = torch.utils.data.get_worker_info().num_workers
            worker_id = torch.utils.data.get_worker_info().id
        else:
            worker_id = 0
            worker_total_num = 1

        worker_trajectories = trajectories[worker_id::worker_total_num]

        for trajectory in worker_trajectories:
            try:
                yield self._process_trajectory(trajectory)
            except Exception as e:
                print(f"Error processing trajectory: {e}")
                continue


class OpenXIterableDataset(IterableDataset):
    """Original OpenX dataset for RLDS format."""

    def __init__(
        self,
        data_root_dir,
        output_dir,
        data_mix,
        image_size=224,
        max_length=1024,
        is_train=True,
        shuffle_buffer_size=1000_000,
        tsfm_thread_muti=1,
        read_thread_muti=1,
        obs_backward_steps=0,
        obs_backward_delta=1,
        action_forward_steps=0,
        use_raw_dataloader=False,
        fix_raw_length=None,
        vla_processor=None,
    ):
        super(OpenXIterableDataset, self).__init__()
        self.data_root_dir = data_root_dir
        self.data_mix = data_mix
        self.use_raw_dataloader = use_raw_dataloader
        self.vla_processor = vla_processor
        self.image_size = image_size
        self.max_length = max_length
        self.is_train = is_train

        self.total_ranks = torch.distributed.get_world_size()
        self.current_rank = torch.distributed.get_rank()

        if self.data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        else:
            mixture_spec = [(os.path.join(self.data_mix, "1.0.0"), 1.0)]
        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_camera_views=("primary",),
            load_depth=False,
            load_proprio=False,
            load_language=True,
            action_proprio_normalization_type=NormalizationType.BOUNDS_Q99,
        )
        self.dataset_num = len(weights)
        self.rlds_config = dict(
            traj_transform_kwargs=dict(
                backward_windows_size=obs_backward_steps,
                backward_delta=obs_backward_delta,
                forward_window_size=action_forward_steps,
                skip_unlabeled=True,
                goal_relabeling_strategy="uniform",
            ),
            frame_transform_kwargs=dict(
                resize_size=(self.image_size, self.image_size),
                num_parallel_calls=16,
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=len(mixture_spec) * tsfm_thread_muti,
            traj_read_threads=len(mixture_spec) * read_thread_muti,
            train=self.is_train,
            shuffle_seed=3407 * self.current_rank,
        )        
        self.rlds_config["frame_transform_kwargs"].update(
            {
                "image_augment_kwargs": dict(
                    random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                    random_brightness=[0.2],
                    random_contrast=[0.8, 1.2],
                    random_saturation=[0.8, 1.2],
                    random_hue=[0.05],
                    augment_order=[
                        "random_resized_crop",
                        "random_brightness",
                        "random_contrast",
                        "random_saturation",
                        "random_hue",
                    ],
                )
            }
        )
        self.rlds_dataset = None
        expected_length, self.ds_stats, self.sample_weights = dataset_statistics(**self.rlds_config)
        self.raw_length = expected_length * self.dataset_num

        if fix_raw_length:
            self.raw_length = fix_raw_length
            print(f"[Dataset] set a fixed dataset length {fix_raw_length} avoids the unexceptable traing interrupt!")

        self.ds_stats_pc = save_dataset_statistics(self.ds_stats, Path(output_dir) / "ds_stats.json")

    def __len__(self):
        if self.use_raw_dataloader:
            return self.raw_length // self.total_ranks
        else:
            return self.raw_length

    def multi_modal_get_item(self, data_item):
        pixel_values_seq = []
        
        for image_primary in data_item["observation"]["image_primary"]:
            image = Image.fromarray(image_primary)
            pixel_values_seq += [image]

        actions = torch.from_numpy(data_item["action"])
        lang = data_item["task"]["language_instruction"].lower()
        if isinstance(lang, bytes): lang = lang.decode()
        
        ret = self.vla_processor(
            text=lang, 
            images=pixel_values_seq,
            suffix_actions=actions,
            return_tensors="pt",
            padding=False,
            max_length=self.max_length,
            truncation=True,
            do_normalize=False,
        )

        model_inputs = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            token_type_ids=ret["token_type_ids"][0],
            attention_mask=ret["attention_mask"][0],
            pixel_values=ret["pixel_values"],
            intrinsic=ret["intrinsic"],
            actions=actions,
        )
        return model_inputs

    def __iter__(self):
        if self.rlds_dataset is None:
            self.rlds_dataset = build_interleaved_dataset(weights=self.sample_weights, dataset_statistics=self.ds_stats, **self.rlds_config).as_numpy_iterator()
            if torch.utils.data.get_worker_info() is not None:
                worker_total_num = torch.utils.data.get_worker_info().num_workers
                worker_id = torch.utils.data.get_worker_info().id
            else:
                worker_id = 0
                worker_total_num = 1
            self.rlds_dataset = itertools.islice(iter(self.rlds_dataset), worker_id, None, worker_total_num)

        for i, data_item in enumerate(self.rlds_dataset):
            ret = self.multi_modal_get_item(data_item)
            if i < len(self):
                yield ret
            else:
                break


def build_datasets(
    data_args,
    output_dir,
    vla_processor=None,
) -> IterableDataset:
    """
    Build datasets supporting both RLDS and Nebula formats.
    """
    
    # Check if using Nebula dataset
    if hasattr(data_args, 'use_nebula_dataset') and data_args.use_nebula_dataset:
        # Nebula dataset configuration
        nebula_task_names = [
            "Control-PlaceSphere-Easy", "Control-PushCube-Easy", "Control-StackCube-Easy", 
            "Control-PegInsertionSide-Medium", "Control-PlaceSphere-Medium", "Control-StackCube-Medium", 
            "Control-PlaceSphere-Hard", "Control-StackCube-Hard", 
            "Perception-PickBiggerSphere-Easy", "Perception-PickRedSphere-Easy", "Perception-PickSphere-Easy", 
            "Perception-PlaceDiffCubes-Medium", "Perception-PlaceRedT-Medium", "Perception-PlaceWhitePeg-Medium", 
            "Perception-PlacePeg-Hard", "Perception-PlaceRedT-Hard", "Perception-PlaceRightCubes-Hard", 
            "DynamicEasy-PressSwitch", "DynamicMedium-PickSlidingCube", "DynamicHard-ColorSwitchPickCube", "DynamicHard-ShapeSwitchPickCube",
            "SpatialReferenceEasy-MoveCube", "SpatialReferenceEasy-PickCube"
        ]

        nebula_task_instruction_map = {
            # control tasks
            "Control-PlaceSphere-Easy": "Pick up the blue sphere and place it into the bin",
            "Control-PushCube-Easy": "Push the cube to the target position",
            "Control-StackCube-Easy": "Stack the cube on top of the other cube",
            "Control-PlaceSphere-Medium": "Pick up the blue sphere and place it into the purple bin, and then place it into the blue bin",
            "Control-PegInsertionSide-Medium": "Pick up the peg and insert the orange end into the box with a hole in it",
            "Control-StackCube-Medium": "Pick up the red cube and place it by the green cube, and then pick up the blue cube and place it on top of the two cubes",
            "Control-PlaceSphere-Hard": "Place a sphere to the red bin, and move it to the blue bin, then move it to the green bin",
            "Control-PlugCharger-Hard": "Pick up the plug and insert it into the correct empty slot",
            "Control-StackCube-Hard": "Pick up the red cube and place it by the green cube, and then pick up the blue cube and place it on top of the two cubes, and then pick up the purple cube and place it by the green cube",
            
            # perception tasks
            "Perception-PickBiggerSphere-Easy": "Place the bigger sphere into the bin",
            "Perception-PickRedSphere-Easy": "Place the red sphere into the bin",
            "Perception-PickSphere-Easy": "Place the sphere into the bin",
            "Perception-PlaceRedT-Medium": "Place the red 'T' into the bin",
            "Perception-PlaceDiffCubes-Medium": "Place the cube that has different size into the bin",
            "Perception-PlaceWhitePeg-Medium": "Place the peg that has white color into the bin",
            "Perception-PlaceRedT-Hard": "Place the red 'T' into the bin",
            "Perception-PlaceRightCubes-Hard": "Place the cube that can fit the bin into the bin",
            "Perception-PlacePeg-Hard": "Place the peg that has red color at the middle into the bin",
            
            # spatial reasoning tasks
            "Spatial-PlaceBetween-Easy": "Place the red cube between the blue and green cube",
            "Spatial-PickClosest-Medium": "Pick the cube which is closest to the red cube",
            "Spatial-BuildBlock-Hard": "Create a three-level tower: red cube at bottom, green cube in middle, blue triangle at top.",
            
            # dynamic tasks
            "Dynamic-PressSwitch-Easy": "Only press the switch after the light turns red",
            "Dynamic-ColorSwitchPickCube-Easy": "Pick up the red cube",
            "Dynamic-ShapeSwitchPickCube-Easy": "Pick up the cube",
            "Dynamic-PlaceRollingSphere-Medium": "Place the sphere into the bin",
            "Dynamic-PickCubeWithCollision-Medium": "Pick up the cube",
            "Dynamic-PickCubeWithSliding-Medium": "Pick up the cube",
            "Dynamic-RollBallWithDistraction-Hard": "Roll the ball to the target region",
            "Dynamic-PlaceRollingSphere-Hard": "Place the rolling sphere into the shallow bin, but only when the light turns green",
            
            # robust tasks
            "Robust-PlaceSphere-Easy": "Pick up the blue sphere and place it into the purple bin, and then place it into the blue bin",
            "Robust-PushCube-Easy": "Push the cube to the target goal position",
            "Robust-StackCube-Easy": "Pick up the red cube and place it by the green cube, and then pick up the blue cube and place it on top of the two cubes, and then pick up the purple cube and place it by the green cube",
            "Robust-PlaceSphere-Medium": "Pick up the yellow sphere and place it into the purple bin, and then place it into the blue bin",
            "Robust-PushCube-Medium": "Push the cube to the target goal position",
            "Robust-StackCube-Medium": "Pick up the yellow cube and place it by the blue cube, and then pick up the red cube and place it on top of the two cubes, and then pick up the green cube and place it by the blue cube",
            "Robust-AssemblingKits-Hard": "Assemble the kit by inserting the peg into the hole",
            "Robust-LiftPegUpright-Hard": "Lift the peg and orient it upright",
            
            # adaptation tasks
            "AdaptationTest-MovingCube": "Pick up the cube",
        }

        train_dataset = NebulaIterableDataset(
            data_root_dir=getattr(data_args, 'nebula_data_root', os.path.expanduser("~/mnt_hpc_data_alpha/nebula")),
            output_dir=output_dir,
            # task_names=nebula_task_names,
            task_names=data_args.nebula_task_list,
            task_instruction_map=nebula_task_instruction_map,
            image_size=getattr(data_args, 'image_size', 224),
            max_length=data_args.max_seq_length,
            is_train=True,
            shuffle_buffer_size=getattr(data_args, 'shuffle_buffer_size', 1000_000),
            obs_backward_steps=getattr(data_args, 'obs_backward_steps', 0),
            obs_backward_delta=getattr(data_args, 'obs_backward_delta', 1),
            action_forward_steps=getattr(data_args, 'action_forward_steps', 0),
            max_trajectories_per_task=getattr(data_args, 'max_trajectories_per_task', None),
            vla_processor=vla_processor,  
        )
        
        eval_dataset = None
        return train_dataset, eval_dataset
    
    else:
        # Original RLDS dataset
        train_dataset = OpenXIterableDataset(
            data_args.data_root_dir,
            output_dir,
            data_args.data_mix,
            is_train=True,
            max_length=data_args.max_seq_length,
            shuffle_buffer_size=data_args.shuffle_buffer_size,
            tsfm_thread_muti=data_args.tsfm_thread_muti,
            read_thread_muti=data_args.read_thread_muti,
            obs_backward_steps=data_args.obs_backward_steps,
            obs_backward_delta=data_args.obs_backward_delta,
            action_forward_steps=data_args.action_forward_steps,
            use_raw_dataloader=data_args.use_raw_dataloader,
            fix_raw_length=data_args.fix_raw_length,
            vla_processor=vla_processor,
        )
        eval_dataset = None
        return train_dataset, eval_dataset