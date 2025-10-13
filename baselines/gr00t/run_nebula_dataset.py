"""
Complete integration example: HDF5 dataset → GR00T training pipeline
"""

from pathlib import Path
from torch.utils.data import DataLoader
from gr00t.data.nebula_dataset import HDF5LeRobotDataset  # Your adapter
from gr00t.experiment.data_config import CustomPandaHDF5DataConfig  # Your config
import PIL
import torch
import numpy as np

def create_panda_dataset(
    dataset_root: str,
    task_name: str,
    subtask_name: str
) -> HDF5LeRobotDataset:
    """Create a Panda robot dataset from HDF5 files."""
    
    # Initialize your custom data config
    data_config = CustomPandaHDF5DataConfig()
    
    # Create the dataset
    dataset = HDF5LeRobotDataset(
        dataset_path=dataset_root,
        task_name=task_name,
        subtask_name=subtask_name,
        modality_configs=data_config.modality_config(),
        embodiment_tag="new_embodiment",  # Custom embodiment
        camera_names=["base_camera", "hand_camera"],  # Match modality.json
        state_keys=["single_arm", "gripper"],  # From modality.json 
        action_dim=8,  # 7 arm + 1 gripper = 8 total DOF
        transforms=data_config.transform(),  # Apply the transform pipeline
    )
    
    return dataset

def custom_collate_fn(batch):
    print(f"DEBUG: Processing batch with {len(batch)} samples")
    
    # Recursively convert PIL Images to tensors
    def convert_pil_images(obj):
        if isinstance(obj, PIL.Image.Image):
            print(f"    Found PIL Image! Converting to tensor...")
            return torch.from_numpy(np.array(obj)).permute(2, 0, 1)  # HWC -> CHW
        elif isinstance(obj, dict):
            return {key: convert_pil_images(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_pil_images(item) for item in obj]
        else:
            return obj
    
    # Process each sample
    for i, sample in enumerate(batch):
        print(f"DEBUG: Sample {i}:")
        for key, value in sample.items():
            print(f"  {key}: {type(value)}")
            if key == 'eagle_content':
                print(f"    Processing eagle_content dict...")
                sample[key] = convert_pil_images(value)
    
    print("DEBUG: About to call default_collate")
    return torch.utils.data.default_collate(batch)

def main():
    # Setup paths
    dataset_root = "/HDD1/embodied_ai/data/Nebula/Nebula-alpha"
    task_name = "Control-PlaceSphere-Easy"
    subtask_name = "0"
    
    # Create dataset
    dataset = create_panda_dataset(dataset_root, task_name, subtask_name)
    
    print(f"Dataset loaded: {len(dataset)} samples")
    print(f"Trajectories: {len(dataset.trajectory_ids)}")
    
    # Create DataLoader for training
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        #collate_fn=custom_collate_fn
    )
    
    # Test data loading
    batch = next(iter(dataloader))
    print("\nBatch structure:")
    for key, value in batch.items():
        if hasattr(value, 'shape'):
            print(f"  {key}: {value.shape} ({value.dtype})")
        else:
            print(f"  {key}: {type(value)}")
    
    # Expected output:
    # video: torch.Size([8, T, V, 224, 224, 3]) (torch.float32)
    # state: torch.Size([8, T, 21]) (torch.float32)  # 7+7+7 concatenated
    # action: torch.Size([8, 16, 7]) (torch.float32)
    # language: List[str] with task descriptions

if __name__ == "__main__":
    main()