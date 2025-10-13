import os
import h5py
import yaml
import numpy as np
# Assuming STATE_VEC_IDX_MAPPING is a dictionary mapping state variable names to indices
from configs.state_vec import STATE_VEC_IDX_MAPPING
import glob
from scipy.interpolate import interp1d
from PIL import Image
import json


def interpolate_action_sequence(action_sequence, target_size):
    """
    Extend the action sequece to `target_size` by linear interpolation.
    
    Args:
        action_sequence (np.ndarray): original action sequence, shape (N, D).
        target_size (int): target sequence length.
    
    Returns:
        extended_sequence (np.ndarray): extended action sequence, shape (target_size, D).
    """
    N, D = action_sequence.shape
    indices_old = np.arange(N)
    indices_new = np.linspace(0, N - 1, target_size)

    interp_func = interp1d(indices_old, action_sequence, 
                           kind='linear', axis=0, assume_sorted=True)
    action_sequence_new = interp_func(indices_new)

    return action_sequence_new


class HDF5VLADataset:
    """
    This class is used to sample episodes from the embodiment dataset
    stored in HDF5 files with nebula dataset structure.
    """
    def __init__(self):
        # The name of your dataset
        self.DATASET_NAME = "nebula"

        with open('../../config.yaml', 'r') as file:
            nebula_config = yaml.safe_load(file)
            dataset_dir = nebula_config['experiment']['dataset_root']

        self.data_dir = dataset_dir
        
        # all tasks in nebula dataset
        self.tasks = []
        self.tasks.extend(config['tasks']['capability'])
        self.tasks.extend(config['tasks']['stress'])

        # Multiple tasks - update with your actual task names
        # self.tasks = ["Control-PlaceSphere-Easy"]
        
        # Load configuration from YAML file
        with open('configs/base.yaml', 'r') as file:
            config = yaml.safe_load(file)
        self.CHUNK_SIZE = config['common']['action_chunk_size']
        self.IMG_HISTORY_SIZE = config['common']['img_history_size']
        self.STATE_DIM = config['common']['state_dim']

        self.num_episode_per_task = 1000
        self.num_subtasks = 10
        self.episodes_per_subtask = 100
        
        # Store data grouped by task and subtask
        self.img = []  # List of [task][subtask][episode] 
        self.state = []  # List of [task][subtask][episode]
        self.action = []  # List of [task][subtask][episode]
        self.task_to_global_idx = {}  # Map (task_idx, subtask_idx, episode_idx) to global index
        
        global_idx = 0
        
        # Load all HDF5 files in memory to speed up data loading
        for task_idx, task in enumerate(self.tasks):
            task_img = []
            task_state = []
            task_action = []
            
            for subtask_idx in range(self.num_subtasks):
                file_dir = os.path.join(self.data_dir, task, 'motionplanning', f'{subtask_idx}')
                file_paths = glob.glob(os.path.join(file_dir, '*.h5'))
                
                if len(file_paths) == 0:
                    print(f"Warning: No .h5 files found in: {file_dir}")
                    continue
                
                file_path = file_paths[0]  # Take the first (and presumably only) .h5 file
                
                with h5py.File(file_path, "r") as f:
                    trajs = list(f.keys())  # traj_0, traj_1, ...
                    # Sort by the traj number
                    trajs = sorted(trajs, key=lambda x: int(x.split('_')[-1]))
                    
                    subtask_img = []
                    subtask_state = []
                    subtask_action = []
                    
                    for episode_idx, traj in enumerate(trajs):
                        # Store mapping for global indexing
                        self.task_to_global_idx[global_idx] = (task_idx, subtask_idx, episode_idx)
                        global_idx += 1
                        
                        # Load images from HDF5
                        images = f[traj]['obs']['sensor_data']['base_camera']['rgb'][:]
                        states = f[traj]['obs']['agent']['qpos'][:]
                        actions = f[traj]['actions'][:]

                        subtask_state.append(states)
                        subtask_action.append(actions)
                        subtask_img.append(images)
                
                task_state.append(subtask_state)
                task_action.append(subtask_action)
                task_img.append(subtask_img)
            
            self.state.append(task_state)
            self.action.append(task_action)
            self.img.append(task_img)

        # Compute normalization statistics across all data
        all_states = []
        all_actions = []
        for task_states in self.state:
            for subtask_states in task_states:
                all_states.extend(subtask_states)
        for task_actions in self.action:
            for subtask_actions in task_actions:
                all_actions.extend(subtask_actions)
        
        self.state_min = np.concatenate(all_states).min(axis=0)
        self.state_max = np.concatenate(all_states).max(axis=0)
        self.action_min = np.concatenate(all_actions).min(axis=0)
        self.action_max = np.concatenate(all_actions).max(axis=0)
        self.action_std = np.concatenate(all_actions).std(axis=0)
        self.action_mean = np.concatenate(all_actions).mean(axis=0)
                    
        self.task2lang = {
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
            "SpatialReasoning-PlaceBetween-Easy": "Place the red cube between the blue and green cube",
            "SpatialReasoning-PickClosest-Medium": "Pick the cube which is closest to the red cube",
            "SpatialReasoning-BuildBlock-Hard": "Create a three-level tower: red cube at bottom, green cube in middle, blue triangle at top.",
            
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
        
    def __len__(self):
        return len(self.tasks) * self.num_episode_per_task

    def get_dataset_name(self):
        return self.DATASET_NAME

    def get_item(self, index=None):
        """
        Get a training sample at a random timestep.

        Args:
            index (int, optional): The index of the episode.
                If not provided, a random episode will be selected.

        Returns:
            sample (dict): A dictionary containing the training sample.
        """
        while True:
            if index is None:
                index = np.random.randint(0, self.__len__())
            valid, sample = self.parse_hdf5_file(index)
            if valid:
                return sample
            else:
                index = np.random.randint(0, self.__len__())

    def parse_hdf5_file(self, index):
        """
        Parse data to generate a training sample at a random timestep.

        Args:
            index (int): Global episode index.

        Returns:
            valid (bool): Whether the episode is valid.
            dict: A dictionary containing the training sample.
        """
        # Map global index to task, subtask, and episode indices
        if index not in self.task_to_global_idx:
            return False, None
            
        task_idx, subtask_idx, episode_idx = self.task_to_global_idx[index]
        
        # Get data for this specific episode
        try:
            images = self.img[task_idx][subtask_idx][episode_idx]
            states = self.state[task_idx][subtask_idx][episode_idx]
            actions = self.action[task_idx][subtask_idx][episode_idx]
        except (IndexError, KeyError):
            return False, None
        
        num_steps = len(actions)
        if num_steps == 0:
            return False, None
            
        step_index = np.random.randint(0, num_steps)

        task_name = self.tasks[task_idx]
            
        
        if task_name in self.lang2task:
            language = self.task2lang[self.tasks[task_idx]]
        else:
            with open(self.data_dir + '/' + task_name + + '/'
                   + str(subtask_idx) + '/metadata.json', 'r') as file:
                metadata = json.load(file)
                language = metadata["episodes"][episode_idx]["task_instruction"]
        
        # Skip invalid episodes for specific tasks (keeping original logic)
        if self.tasks[task_idx] == 'PegInsertionSide-v1':
            global_episode_in_task = subtask_idx * self.episodes_per_subtask + episode_idx
            if global_episode_in_task > 400:
                return False, None
        
        # Normalize states and actions to [-1, 1]
        states_norm = (states - self.state_min) / (self.state_max - self.state_min) * 2 - 1
        states_norm = states_norm[:, :-1]  # Remove the last state as it is replicate of the -2 state
        actions_norm = (actions - self.action_min) / (self.action_max - self.action_min) * 2 - 1
        
        # Get image history from HDF5 data
        start_img_idx = max(0, step_index - self.IMG_HISTORY_SIZE + 1)
        end_img_idx = step_index + 1
        img_history = images[start_img_idx:end_img_idx]
        img_valid_len = img_history.shape[0]

        # Pad images if necessary
        if img_valid_len < self.IMG_HISTORY_SIZE:
            padding = np.tile(img_history[0:1], (self.IMG_HISTORY_SIZE - img_valid_len, 1, 1, 1))
            img_history = np.concatenate([padding, img_history], axis=0)

        img_history_mask = np.array(
            [False] * (self.IMG_HISTORY_SIZE - img_valid_len) + [True] * img_valid_len
        )

        # Compute state statistics
        state_std = np.std(states_norm, axis=0)
        state_mean = np.mean(states_norm, axis=0)
        state_norm = np.sqrt(np.mean(states_norm ** 2, axis=0))

        # Get state and action at the specified timestep
        state = states_norm[step_index: step_index + 1]
        runtime_chunksize = self.CHUNK_SIZE // 4
        action_sequence = actions_norm[step_index: step_index + runtime_chunksize]

        # Pad action sequence if necessary
        if action_sequence.shape[0] < runtime_chunksize:
            padding = np.tile(action_sequence[-1:], (runtime_chunksize - action_sequence.shape[0], 1))
            action_sequence = np.concatenate([action_sequence, padding], axis=0)

        action_sequence = interpolate_action_sequence(action_sequence, self.CHUNK_SIZE)

        # Fill state and action into unified vectors
        def fill_in_state(values):
            UNI_STATE_INDICES = [
                STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"] for i in range(7)
            ] + [
                STATE_VEC_IDX_MAPPING[f"right_gripper_open"]
            ]
            uni_vec = np.zeros(values.shape[:-1] + (self.STATE_DIM,))
            uni_vec[..., UNI_STATE_INDICES] = values
            return uni_vec

        state_indicator = fill_in_state(np.ones_like(state_std))
        state = fill_in_state(state)
        state_std = fill_in_state(state_std)
        state_mean = fill_in_state(state_mean)
        state_norm_vec = fill_in_state(state_norm)
        action_sequence = fill_in_state(action_sequence)

        # Assemble the meta information
        meta = {
            "dataset_name": self.DATASET_NAME,
            "#steps": num_steps,
            "step_id": step_index,
            "instruction": language,
            "task_idx": task_idx,
            "subtask_idx": subtask_idx,
            "episode_idx": episode_idx
        }

        # Return the resulting sample
        return True, {
            "meta": meta,
            "state": state,
            "state_std": state_std,
            "state_mean": state_mean,
            "state_norm": state_norm_vec,
            "actions": action_sequence,
            "state_indicator": state_indicator,
            "cam_high": img_history,  # Images from base_camera
            "cam_high_mask": img_history_mask,
            "cam_left_wrist": np.zeros((self.IMG_HISTORY_SIZE, 0, 0, 0)),
            "cam_left_wrist_mask": np.zeros(self.IMG_HISTORY_SIZE, dtype=bool),
            "cam_right_wrist": np.zeros((self.IMG_HISTORY_SIZE, 0, 0, 0)),
            "cam_right_wrist_mask": np.zeros(self.IMG_HISTORY_SIZE, dtype=bool),
        }


if __name__ == "__main__":
    ds = HDF5VLADataset()

    json_data = {
        'state_min': ds.state_min.tolist(),
        'state_max': ds.state_max.tolist(),
        'action_min': ds.action_min.tolist(),
        'action_max': ds.action_max.tolist(),
    }
    print(json_data)

    # Test loading a sample
    sample = ds.get_item(0)
    print("Sample keys:", sample.keys())
    print("Meta info:", sample['meta'])
    print("Image shape:", sample['cam_high'].shape)
    print("State shape:", sample['state'].shape)
    print("Action shape:", sample['actions'].shape)