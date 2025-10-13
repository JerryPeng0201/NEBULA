"""
Modified script to load HDF5 datasets for GR00T training pipeline
"""

import json
import pathlib
import time
from dataclasses import dataclass, field
from pprint import pprint
from typing import List, Literal

import matplotlib.pyplot as plt
import numpy as np
import tyro

import sys
sys.path.append('./Isaac-GR00T')

# Import your custom dataset and config
from gr00t.data.nebula_dataset import HDF5LeRobotDataset
from gr00t.experiment.data_config import CustomPandaHDF5DataConfig
from gr00t.data.embodiment_tags import EMBODIMENT_TAG_MAPPING, EmbodimentTag
from gr00t.utils.misc import any_describe


def print_yellow(text: str) -> None:
    """Print text in yellow color"""
    print(f"\033[93m{text}\033[0m")


@dataclass
class HDF5ArgsConfig:
    """Configuration for loading the HDF5 dataset."""

    dataset_root: str = "/HDD1/embodied_ai/data/Nebula/Nebula-alpha"
    """Root path to the dataset."""
    
    task_name: str = "Control-PlaceSphere-Easy"
    """Task name folder."""
    
    subtask_name: str = "0"
    """Subtask folder name."""

    embodiment_tag: Literal[tuple(EMBODIMENT_TAG_MAPPING.keys())] = "new_embodiment"
    """Embodiment tag to use."""

    video_backend: Literal["decord", "torchvision_av"] = "torchvision_av"
    """Backend to use for video loading."""

    plot_state_action: bool = False
    """Whether to plot the state and action space."""

    steps: int = 200
    """Number of steps to plot."""


#####################################################################################


def get_modality_keys_from_json(dataset_path: pathlib.Path) -> dict[str, list[str]]:
    """
    Get the modality keys from the modality.json file in the HDF5 dataset.
    Returns a dictionary with modality types as keys and their corresponding modality keys as values.
    """
    modality_path = dataset_path / "modality.json"
    if not modality_path.exists():
        # Return default structure for Panda robot
        return {
            "video": ["video.base_view", "video.hand_view", "video.wrist_view"],
            "state": ["state.single_arm", "state.gripper"], 
            "action": ["action.single_arm", "action.gripper"],
            "annotation": ["annotation.human.action.task_description"]
        }
    
    with open(modality_path, "r") as f:
        modality_meta = json.load(f)

    # Convert modality structure to the expected format
    modality_dict = {}
    for key in modality_meta.keys():
        modality_dict[key] = []
        for modality in modality_meta[key]:
            modality_dict[key].append(f"{key}.{modality}")
    return modality_dict


def plot_state_action_space(
    state_dict: dict[str, np.ndarray],
    action_dict: dict[str, np.ndarray],
    shared_keys: list[str] = ["single_arm", "gripper"],  # Updated for Panda
):
    """
    Plot the state and action space side by side.

    state_dict: dict[str, np.ndarray] with key: [Time, Dimension]
    action_dict: dict[str, np.ndarray] with key: [Time, Dimension]
    shared_keys: list[str] of keys to plot (without the "state." or "action." prefix)
    """
    # Create a figure with one subplot per shared key
    fig = plt.figure(figsize=(16, 4 * len(shared_keys)))

    # Create GridSpec to organize the layout
    gs = fig.add_gridspec(len(shared_keys), 1)

    # Color palette for different dimensions
    colors = plt.cm.tab10.colors

    for i, key in enumerate(shared_keys):
        state_key = f"state.{key}"
        action_key = f"action.{key}"

        # Skip if either key is not in the dictionaries
        if state_key not in state_dict or action_key not in action_dict:
            print(
                f"Warning: Skipping {key} as it's not found in both state and action dictionaries"
            )
            continue

        # Get the data
        state_data = state_dict[state_key]
        action_data = action_dict[action_key]

        print(f"{state_key}.shape: {state_data.shape}")
        print(f"{action_key}.shape: {action_data.shape}")

        # Create subplot
        ax = fig.add_subplot(gs[i, 0])

        # Plot each dimension with a different color
        # Determine the minimum number of dimensions to plot
        min_dims = min(state_data.shape[1], action_data.shape[1])

        for dim in range(min_dims):
            # Create time arrays for both state and action
            state_time = np.arange(len(state_data))
            action_time = np.arange(len(action_data))

            # State with dashed line
            ax.plot(
                state_time,
                state_data[:, dim],
                "--",
                color=colors[dim % len(colors)],
                linewidth=1.5,
                label=f"state dim {dim}",
            )

            # Action with solid line (same color as corresponding state dimension)
            ax.plot(
                action_time,
                action_data[:, dim],
                "-",
                color=colors[dim % len(colors)],
                linewidth=2,
                label=f"action dim {dim}",
            )

        ax.set_title(f"{key}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.grid(True, linestyle=":", alpha=0.7)

        # Create a more organized legend
        handles, labels = ax.get_legend_handles_labels()
        # Sort the legend so state and action for each dimension are grouped
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc="upper right")

    plt.tight_layout()


def plot_image(image: np.ndarray):
    """
    Plot the image.
    """
    # matplotlib show the image
    plt.imshow(image)
    plt.axis("off")
    plt.pause(0.05)  # Non-blocking show
    plt.clf()  # Clear the figure for the next frame


def load_hdf5_dataset(
    dataset_root: str,
    task_name: str,
    subtask_name: str,
    embodiment_tag: str,
    video_backend: str = "decord",
    steps: int = 200,
    plot_state_action: bool = True,
):
    # 1. Build full dataset path
    dataset_path = pathlib.Path(dataset_root) / task_name / "motionplanning" / subtask_name
    print(f"Looking for dataset at: {dataset_path}")
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    # 2. Get modality keys from modality.json or use defaults
    modality_keys_dict = get_modality_keys_from_json(dataset_path)
    video_modality_keys = modality_keys_dict["video"]
    language_modality_keys = modality_keys_dict.get("annotation", [])
    state_modality_keys = modality_keys_dict["state"]
    action_modality_keys = modality_keys_dict["action"]

    pprint(f"Valid modality_keys: {modality_keys_dict} \n")

    print(f"state_modality_keys: {state_modality_keys}")
    print(f"action_modality_keys: {action_modality_keys}")

    # 3. Initialize your custom data config
    data_config = CustomPandaHDF5DataConfig()

    # 4. Create the HDF5 dataset
    print(f"Loading HDF5 dataset from {dataset_root}")
    
    dataset = HDF5LeRobotDataset(
        dataset_path=dataset_root,
        task_name=task_name,
        subtask_name=subtask_name,
        modality_configs=data_config.modality_config(),
        embodiment_tag=embodiment_tag,
        camera_names=["base_camera", "hand_camera"],  # Match your modality.json
        state_keys=["state.single_arm", "state.gripper"],  # Remove "state." prefix
        action_dim=8,  # Based on your modality.json: 7 arm + 1 gripper
        transforms=data_config.transform(),  # Apply the transform pipeline
        video_backend=video_backend,
    )

    print("\n" * 2)
    print("=" * 100)
    print(f"{' HDF5 Panda Dataset ':=^100}")
    print("=" * 100)

    # Test loading a single data point
    print(f"Dataset loaded successfully!")
    print(f"Total samples: {len(dataset)}")
    print(f"Total trajectories: {len(dataset.trajectory_ids)}")

    # Get the first data point
    resp = dataset[0]
    any_describe(resp)
    print(f"Data keys: {resp.keys()}")

    print("=" * 50)
    for key, value in resp.items():
        if isinstance(value, np.ndarray):
            print(f"{key}: {value.shape} ({value.dtype})")
        elif hasattr(value, 'shape'):  # torch.Tensor
            print(f"{key}: {value.shape} ({value.dtype})")
        else:
            print(f"{key}: {value}")

    # 5. Collect data for visualization
    images_list = []
    video_key = video_modality_keys[0]  # Use the first video modality

    state_dict = {key: [] for key in state_modality_keys}
    action_dict = {key: [] for key in action_modality_keys}

    total_images = 20  # show 20 images
    skip_frames = steps // total_images

    print(f"\nCollecting {steps} samples for visualization...")
    
    for i in range(min(steps, len(dataset))):
        try:
            resp = dataset[i]
    
            if i % skip_frames == 0:
                # Handle GR00T transformed format
                if 'eagle_content' in resp and 'image_inputs' in resp['eagle_content']:
                    # Extract image from eagle_content format
                    images = resp['eagle_content']['image_inputs']
                    if images and len(images) > 0:
                        img = images[0]  # First image
                        print(f"Image {i}, eagle format")
                        images_list.append(img.copy())
                
                # Handle language data from eagle_content
                if 'eagle_content' in resp and 'text_list' in resp['eagle_content']:
                    text_data = resp['eagle_content']['text_list']
                    if text_data:
                        print(f"Text prompt: {text_data[0]}...")  # First 100 chars

            # Collect state and action data
            # For GR00T format, we have concatenated state and action
            if "state" in resp:
                # resp["state"] is concatenated: [1, 64] - all states together
                state_data = resp["state"]
                if hasattr(state_data, 'numpy'):
                    combined_state = state_data[0].numpy()  # Shape: [64]
                else:
                    combined_state = state_data[0]
                
                # Since we can't split back to individual modalities easily,
                # just store the combined state
                if "combined_state" not in state_dict:
                    state_dict["combined_state"] = []
                state_dict["combined_state"].append(combined_state)
            
            if "action" in resp:
                # resp["action"] is concatenated: [16, 32] - all actions together
                action_data = resp["action"]
                if hasattr(action_data, 'numpy'):
                    combined_action = action_data[0].numpy()  # Shape: [32]
                else:
                    combined_action = action_data[0]
                    
                if "combined_action" not in action_dict:
                    action_dict["combined_action"] = []
                action_dict["combined_action"].append(combined_action)
                    
        except Exception as e:
            print(f"Error loading sample {i}: {e}")
            break
            
        if i % 50 == 0:
            print(f"Processed {i}/{steps} samples...")

    # Convert lists to numpy arrays
    print("Converting data to numpy arrays...")
    for state_key in state_modality_keys:
        if state_dict[state_key]:
            state_dict[state_key] = np.array(state_dict[state_key])
            print(f"Final {state_key} shape: {state_dict[state_key].shape}")
            
    for action_key in action_modality_keys:
        if action_dict[action_key]:
            action_dict[action_key] = np.array(action_dict[action_key])
            print(f"Final {action_key} shape: {action_dict[action_key].shape}")

    # 6. Plot state and action space
    if plot_state_action and state_dict and action_dict:
        print("Plotting state and action space...")
        plot_state_action_space(state_dict, action_dict, shared_keys=["single_arm", "gripper"])
        plt.show()

    # 7. Save sample images instead of plotting (for server use)
    if images_list:
        print(f"Saving {len(images_list)} sample images...")
        
        # Create output directory
        import os
        output_dir = "sample_images"
        os.makedirs(output_dir, exist_ok=True)
        
        for i, img in enumerate(images_list):
            # Handle PIL Image objects directly
            if hasattr(img, 'save'):  # PIL Image
                img.save(f"{output_dir}/sample_{i*skip_frames:03d}.png")
            else:  # numpy array (shouldn't happen with eagle_content but just in case)
                from PIL import Image
                if isinstance(img, np.ndarray):
                    pil_img = Image.fromarray(img.astype(np.uint8))
                    pil_img.save(f"{output_dir}/sample_{i*skip_frames:03d}.png")
        
        print(f"Images saved to {output_dir}/ directory")
    else:
        print("No images collected for saving")

    print("Dataset loading and visualization completed!")


if __name__ == "__main__":
    config = tyro.cli(HDF5ArgsConfig)
    load_hdf5_dataset(
        config.dataset_root,
        config.task_name,
        config.subtask_name,
        config.embodiment_tag,
        config.video_backend,
        config.steps,
        config.plot_state_action,
    )