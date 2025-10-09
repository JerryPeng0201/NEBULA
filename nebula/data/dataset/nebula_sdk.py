"""
NEBULA Python SDK - Clean API interface for robotics datasets
"""

from typing import List, Optional, Dict, Any, Iterator, Union
from pathlib import Path
import warnings
from dataclasses import dataclass

from nebula.dataset.nebula_dataset import NEBULADataset
from nebula.dataset.embodiment import Embodiment
from nebula.dataset.structures import Episode, Step, Observation, Action


@dataclass
class DatasetConfig:
    """Configuration for dataset loading."""
    dataset_root: Union[str, Path]
    robot_config_path: Union[str, Path] 
    robot_name: str
    task_filter: Optional[List[str]] = None
    load_statistics: bool = True


class EpisodeQuery:
    """Query builder for filtering episodes."""
    
    def __init__(self, dataset: 'NEBULADatasetSDK'):
        self._dataset = dataset
        self._filters = {}
        
    def task(self, task_name: str) -> 'EpisodeQuery':
        """Filter by task name."""
        self._filters['task'] = task_name
        return self
        
    def success(self, success: bool) -> 'EpisodeQuery':
        """Filter by success status."""
        self._filters['success'] = success
        return self
        
    def robot(self, robot_name: str) -> 'EpisodeQuery':
        """Filter by robot name.""" 
        self._filters['robot'] = robot_name
        return self
        
    def subtask(self, task_name: str, subtask_idx: str) -> 'EpisodeQuery':
        """Filter by specific subtask."""
        self._filters['task'] = task_name
        self._filters['subtask'] = subtask_idx
        return self
        
    def instruction_contains(self, text: str) -> 'EpisodeQuery':
        """Filter by instruction text content."""
        self._filters['instruction_contains'] = text.lower()
        return self
        
    def min_steps(self, steps: int) -> 'EpisodeQuery':
        """Filter by minimum number of steps."""
        self._filters['min_steps'] = steps
        return self
        
    def max_steps(self, steps: int) -> 'EpisodeQuery':
        """Filter by maximum number of steps."""
        self._filters['max_steps'] = steps
        return self
        
    def limit(self, limit: int) -> 'EpisodeQuery':
        """Limit number of results."""
        self._filters['limit'] = limit
        return self
        
    def offset(self, offset: int) -> 'EpisodeQuery':
        """Skip first N results."""
        self._filters['offset'] = offset
        return self
        
    def execute(self) -> List[Episode]:
        """Execute the query and return episodes."""
        return self._dataset._execute_query(self._filters)
        
    def count(self) -> int:
        """Count episodes matching the query."""
        episodes = self.execute()
        return len(episodes)
        
    def first(self) -> Optional[Episode]:
        """Get first episode matching the query."""
        episodes = self.limit(1).execute()
        return episodes[0] if episodes else None
        
    def __iter__(self) -> Iterator[Episode]:
        """Iterate over matching episodes."""
        for episode in self.execute():
            yield episode


class NEBULADatasetSDK:
    """
    High-level API for accessing NEBULA unified robotics datasets.
    
    This is the main entry point for the NEBULA SDK. It provides a clean,
    intuitive interface for loading and querying robotics datasets.
    
    Example:
        # Load a dataset
        dataset = NEBULADataset.load_nebula(
            dataset_root="/path/to/nebula",
            robot_config="franka_panda_single_arm_2gripper"
        )
        
        # Query episodes
        successful_episodes = dataset.episodes().success(True).execute()
        
        # Get specific episode
        episode = dataset.get_episode(0)
        
        # Iterate through episodes
        for episode in dataset.episodes().task("pick_and_place"):
            print(episode.instruction.raw_text)
    """
    
    def __init__(self, dataset_impl: NEBULADataset):
        """Initialize with a dataset implementation."""
        self._dataset = dataset_impl
        self._cache = {}
        
    @classmethod
    def load_nebula(cls, 
                   dataset_root: Union[str, Path],
                   robot_config_path: Union[str, Path] = "./nebula/dataset/embodiment_configs.py",
                   robot_name: str = "franka_panda_single_arm_2gripper",
                   task_filter: Optional[List[str]] = None,
                   load_statistics: bool = True) -> 'NEBULADatasetSDK':
        """
        Load a NEBULA dataset.
        
        Args:
            dataset_root: Path to NEBULA dataset root directory
            robot_config_path: Path to robot configuration file
            robot_name: Robot configuration to use
            task_filter: Optional list of tasks to include
            load_statistics: Whether to compute episode statistics
            
        Returns:
            NEBULADatasetSDK instance
            
        Raises:
            FileNotFoundError: If dataset or config files not found
            ValueError: If robot configuration is invalid
        """
        try:
            dataset = NEBULADataset(
                dataset_root=dataset_root,
                robot_config_path=robot_config_path,
                robot_name=robot_name,
                task_filter=task_filter,
                load_statistics=load_statistics
            )
            return cls(dataset)
        except Exception as e:
            raise ValueError(f"Failed to load NEBULA dataset: {e}") from e
    
    @classmethod 
    def load_from_config(cls, config: DatasetConfig) -> 'NEBULADataset':
        """Load dataset from configuration object."""
        return cls.load_nebula(**config.__dict__)
        
    def __len__(self) -> int:
        """Get total number of episodes."""
        return len(self._dataset)
        
    def __getitem__(self, idx: int) -> Episode:
        """Get episode by index."""
        return self.get_episode(idx)
        
    def __iter__(self) -> Iterator[Episode]:
        """Iterate over all episodes."""
        for i in range(len(self)):
            yield self.get_episode(i)
            
    def get_episode(self, idx: int) -> Episode:
        """
        Get a specific episode by index.
        
        Args:
            idx: Episode index
            
        Returns:
            Episode object
            
        Raises:
            IndexError: If index is out of range
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Episode index {idx} out of range [0, {len(self)})")
            
        # Simple caching for recently accessed episodes
        if idx not in self._cache:
            if len(self._cache) > 100:  # Limit cache size
                self._cache.clear()
            self._cache[idx] = self._dataset[idx]
            
        return self._cache[idx]
        
    def episodes(self) -> EpisodeQuery:
        """
        Create a new episode query.
        
        Returns:
            EpisodeQuery builder for filtering episodes
            
        Example:
            # Get successful pick and place episodes
            episodes = dataset.episodes().task("pick_and_place").success(True).execute()
            
            # Get first failed episode with more than 50 steps  
            episode = dataset.episodes().success(False).min_steps(50).first()
        """
        return EpisodeQuery(self)
        
    def get_tasks(self) -> List[str]:
        """Get list of all task names in the dataset."""
        return self._dataset.get_task_names()
        
    def get_robot_info(self) -> Dict[str, Any]:
        """Get robot configuration information."""
        config = self._dataset.robot_config
        return {
            "robot_name": config.robot_name,
            "robot_type": config.robot_type,
            "is_dual_arm": config.is_dual_arm,
            "limbs": config.get_all_limb_names(),
            "state_limbs": config.get_all_state_limbs(),
            "action_limbs": config.get_all_action_limbs(),
            "camera_views": config.get_all_view_names(),
            "total_state_dim": config.get_total_state_dim(),
            "total_action_dim": config.get_total_action_dim_from_modality()
        }
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        return self._dataset.get_dataset_statistics()
        
    def sample(self, n: int = 1, random_state: Optional[int] = None) -> List[Episode]:
        """
        Randomly sample episodes from the dataset.
        
        Args:
            n: Number of episodes to sample
            random_state: Random seed for reproducibility
            
        Returns:
            List of sampled episodes
        """
        import random
        if random_state is not None:
            random.seed(random_state)
            
        indices = random.sample(range(len(self)), min(n, len(self)))
        return [self.get_episode(i) for i in indices]
        
    def train_test_split(self, 
                        test_size: float = 0.2,
                        random_state: Optional[int] = None,
                        stratify_by_task: bool = True) -> tuple['NEBULADatasetSDK', 'NEBULADatasetSDK']:
        """
        Split dataset into train and test sets.
        
        Args:
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
            stratify_by_task: Whether to stratify by task name
            
        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        import random
        if random_state is not None:
            random.seed(random_state)
            
        if stratify_by_task:
            # Split each task separately
            train_indices = []
            test_indices = []
            
            for task in self.get_tasks():
                task_episodes = self._dataset.get_episodes_by_task(task)
                n_test = int(len(task_episodes) * test_size)
                
                test_task_indices = random.sample(task_episodes, n_test)
                train_task_indices = [i for i in task_episodes if i not in test_task_indices]
                
                train_indices.extend(train_task_indices)
                test_indices.extend(test_task_indices)
        else:
            # Simple random split
            all_indices = list(range(len(self)))
            n_test = int(len(all_indices) * test_size)
            test_indices = random.sample(all_indices, n_test)
            train_indices = [i for i in all_indices if i not in test_indices]
            
        # Create subset datasets
        train_dataset = NEBULADatasetSubset(self, train_indices)
        test_dataset = NEBULADatasetSubset(self, test_indices)
        
        return train_dataset, test_dataset
        
    def _execute_query(self, filters: Dict[str, Any]) -> List[Episode]:
        """Execute a query with filters."""
        # Start with all episode indices
        indices = list(range(len(self)))
        
        # Apply task filter
        if 'task' in filters:
            task_indices = self._dataset.get_episodes_by_task(filters['task'])
            indices = [i for i in indices if i in task_indices]
            
        # Apply subtask filter  
        if 'subtask' in filters:
            subtask_indices = self._dataset.get_episodes_by_subtask(
                filters['task'], filters['subtask']
            )
            indices = [i for i in indices if i in subtask_indices]
            
        # Apply other filters by loading episodes
        filtered_episodes = []
        
        for idx in indices:
            try:
                episode = self.get_episode(idx)
                
                # Success filter
                if 'success' in filters:
                    if episode.meta.success != filters['success']:
                        continue
                        
                # Robot filter
                if 'robot' in filters:
                    if episode.robot_name != filters['robot']:
                        continue
                        
                # Instruction content filter
                if 'instruction_contains' in filters:
                    if filters['instruction_contains'] not in episode.instruction.raw_text.lower():
                        continue
                        
                # Step count filters
                if 'min_steps' in filters:
                    if len(episode.steps) < filters['min_steps']:
                        continue
                        
                if 'max_steps' in filters:
                    if len(episode.steps) > filters['max_steps']:
                        continue
                        
                filtered_episodes.append(episode)
                
            except Exception as e:
                warnings.warn(f"Error loading episode {idx}: {e}")
                continue
                
        # Apply offset and limit
        if 'offset' in filters:
            filtered_episodes = filtered_episodes[filters['offset']:]
            
        if 'limit' in filters:
            filtered_episodes = filtered_episodes[:filters['limit']]
            
        return filtered_episodes


class NEBULADatasetSubset(NEBULADatasetSDK):
    """A subset of an NEBULA dataset with specific indices."""
    
    def __init__(self, parent_dataset: NEBULADatasetSDK, indices: List[int]):
        self._parent = parent_dataset
        self._indices = indices
        
    def __len__(self) -> int:
        return len(self._indices)
        
    def __getitem__(self, idx: int) -> Episode:
        if idx < 0 or idx >= len(self._indices):
            raise IndexError(f"Index {idx} out of range [0, {len(self._indices)})")
        return self._parent.get_episode(self._indices[idx])
        
    def get_episode(self, idx: int) -> Episode:
        return self[idx]
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics for this subset."""
        stats = self._parent.get_statistics().copy()
        stats['total_episodes'] = len(self)
        stats['subset'] = True
        return stats


# Convenience functions for common use cases
def load_nebula_dataset(dataset_root: Union[str, Path], 
                       robot_name: str = "franka_panda_single_arm_2gripper",
                       **kwargs) -> NEBULADatasetSDK:
    """
    Quick function to load a NEBULA dataset with minimal configuration.
    
    Args:
        dataset_root: Path to NEBULA dataset
        robot_name: Robot configuration to use
        **kwargs: Additional arguments for dataset loading
        
    Returns:
        NEBULADatasetSDK instance
    """
    return NEBULADatasetSDK.load_nebula(
        dataset_root=dataset_root,
        robot_name=robot_name,
        **kwargs
    )


def list_available_robots(config_path: Union[str, Path] = "./nebula/dataset/embodiment_configs.py") -> List[str]:
    """
    List all available robot configurations.
    
    Args:
        config_path: Path to robot configuration file
        
    Returns:
        List of robot configuration names
    """
    return Embodiment.list_available_robots(config_path)


# Export main classes and functions
__all__ = [
    'NEBULADatasetSDK',
    'EpisodeQuery', 
    'DatasetConfig',
    'load_nebula_dataset',
    'list_available_robots'
]