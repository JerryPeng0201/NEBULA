"""
Common Transform Functions for Robotics Datasets
Provides a library of commonly used transforms for Episode-level data processing.
"""

import numpy as np
import cv2
from typing import Optional, List, Dict, Any, Callable, Tuple
from copy import deepcopy
import random

from nebula.dataset.structures import Episode, Step, Observation, Action, RobotLimbState


# ============================================================================
# Data Filtering Transforms
# ============================================================================

class FilterBySuccess:
    """Filter episodes based on success status."""
    
    def __init__(self, success_only: bool = True):
        self.success_only = success_only
    
    def __call__(self, episode: Episode) -> Optional[Episode]:
        if self.success_only and not episode.meta.success:
            return None  # Filter out failed episodes
        elif not self.success_only and episode.meta.success:
            return None  # Filter out successful episodes
        return episode


class FilterByTask:
    """Filter episodes by task name."""
    
    def __init__(self, allowed_tasks: List[str]):
        self.allowed_tasks = allowed_tasks
    
    def __call__(self, episode: Episode) -> Optional[Episode]:
        if not any(task in episode.task_id for task in self.allowed_tasks):
            return None
        return episode


class FilterByLength:
    """Filter episodes by trajectory length."""
    
    def __init__(self, min_steps: int = 0, max_steps: int = float('inf')):
        self.min_steps = min_steps
        self.max_steps = max_steps
    
    def __call__(self, episode: Episode) -> Optional[Episode]:
        num_steps = len(episode.steps)
        if num_steps < self.min_steps or num_steps > self.max_steps:
            return None
        return episode


class FilterByInstruction:
    """Filter episodes by instruction content."""
    
    def __init__(self, required_words: List[str] = None, forbidden_words: List[str] = None):
        self.required_words = [w.lower() for w in (required_words or [])]
        self.forbidden_words = [w.lower() for w in (forbidden_words or [])]
    
    def __call__(self, episode: Episode) -> Optional[Episode]:
        instruction = episode.instruction.raw_text.lower()
        
        # Check required words
        if self.required_words:
            if not any(word in instruction for word in self.required_words):
                return None
        
        # Check forbidden words
        if self.forbidden_words:
            if any(word in instruction for word in self.forbidden_words):
                return None
        
        return episode


# ============================================================================
# Data Truncation/Subsampling Transforms
# ============================================================================

class TruncateEpisode:
    """Truncate episodes to a maximum length."""
    
    def __init__(self, max_steps: int, truncate_from: str = 'end'):
        """
        Args:
            max_steps: Maximum number of steps to keep
            truncate_from: 'end', 'start', or 'random'
        """
        self.max_steps = max_steps
        self.truncate_from = truncate_from
    
    def __call__(self, episode: Episode) -> Episode:
        if len(episode.steps) <= self.max_steps:
            return episode
        
        episode = deepcopy(episode)
        
        if self.truncate_from == 'end':
            episode.steps = episode.steps[:self.max_steps]
        elif self.truncate_from == 'start':
            episode.steps = episode.steps[-self.max_steps:]
        elif self.truncate_from == 'random':
            start_idx = random.randint(0, len(episode.steps) - self.max_steps)
            episode.steps = episode.steps[start_idx:start_idx + self.max_steps]
        
        return episode


class SubsampleSteps:
    """Subsample steps from episode."""
    
    def __init__(self, step_size: int = 2, keep_first_last: bool = True):
        """
        Args:
            step_size: Take every nth step
            keep_first_last: Always keep first and last steps
        """
        self.step_size = step_size
        self.keep_first_last = keep_first_last
    
    def __call__(self, episode: Episode) -> Episode:
        if len(episode.steps) <= 2:
            return episode
        
        episode = deepcopy(episode)
        
        if self.keep_first_last:
            # Keep first, subsample middle, keep last
            middle_steps = episode.steps[1:-1:self.step_size]
            episode.steps = [episode.steps[0]] + middle_steps + [episode.steps[-1]]
        else:
            episode.steps = episode.steps[::self.step_size]
        
        return episode


class ExtractSubsequence:
    """Extract a random subsequence from the episode."""
    
    def __init__(self, subsequence_length: int, random_start: bool = True):
        self.subsequence_length = subsequence_length
        self.random_start = random_start
    
    def __call__(self, episode: Episode) -> Episode:
        if len(episode.steps) <= self.subsequence_length:
            return episode
        
        episode = deepcopy(episode)
        
        if self.random_start:
            max_start = len(episode.steps) - self.subsequence_length
            start_idx = random.randint(0, max_start)
        else:
            start_idx = 0
        
        episode.steps = episode.steps[start_idx:start_idx + self.subsequence_length]
        return episode


# ============================================================================
# Data Augmentation Transforms  
# ============================================================================

class AddNoiseToRobotState:
    """Add Gaussian noise to robot joint positions/velocities."""
    
    def __init__(self, 
                 position_noise_std: float = 0.01,
                 velocity_noise_std: float = 0.05,
                 limbs: Optional[List[str]] = None):
        self.position_noise_std = position_noise_std
        self.velocity_noise_std = velocity_noise_std
        self.limbs = limbs
    
    def __call__(self, episode: Episode) -> Episode:
        episode = deepcopy(episode)
        
        for step in episode.steps:
            for limb_name, limb_state in step.observation.robot_state.limbs.items():
                if self.limbs is None or limb_name in self.limbs:
                    # Add noise to positions
                    if self.position_noise_std > 0:
                        noise = np.random.normal(0, self.position_noise_std, limb_state.joint_positions.shape)
                        limb_state.joint_positions = limb_state.joint_positions + noise
                    
                    # Add noise to velocities
                    if limb_state.joint_velocities is not None and self.velocity_noise_std > 0:
                        noise = np.random.normal(0, self.velocity_noise_std, limb_state.joint_velocities.shape)
                        limb_state.joint_velocities = limb_state.joint_velocities + noise
        
        return episode


class AddNoiseToActions:
    """Add Gaussian noise to actions."""
    
    def __init__(self, noise_std: float = 0.01, limbs: Optional[List[str]] = None):
        self.noise_std = noise_std
        self.limbs = limbs
    
    def __call__(self, episode: Episode) -> Episode:
        episode = deepcopy(episode)
        
        for step in episode.steps:
            if step.action.robot_action:
                for limb_name, action in step.action.robot_action.limbs.items():
                    if self.limbs is None or limb_name in self.limbs:
                        noise = np.random.normal(0, self.noise_std, action.shape)
                        step.action.robot_action.limbs[limb_name] = action + noise
        
        return episode


class TimeShift:
    """Shift robot states by a small time offset."""
    
    def __init__(self, max_shift_steps: int = 2):
        self.max_shift_steps = max_shift_steps
    
    def __call__(self, episode: Episode) -> Episode:
        if len(episode.steps) <= self.max_shift_steps * 2:
            return episode
        
        episode = deepcopy(episode)
        shift = random.randint(-self.max_shift_steps, self.max_shift_steps)
        
        if shift > 0:
            # Use later observations with current actions
            for i in range(len(episode.steps) - shift):
                episode.steps[i].observation = episode.steps[i + shift].observation
        elif shift < 0:
            # Use earlier observations with current actions  
            for i in range(-shift, len(episode.steps)):
                episode.steps[i].observation = episode.steps[i + shift].observation
        
        return episode


class ImageAugmentation:
    """Apply image augmentations to camera observations."""
    
    def __init__(self, 
                 brightness_range: Tuple[float, float] = (0.8, 1.2),
                 contrast_range: Tuple[float, float] = (0.8, 1.2),
                 add_blur: bool = False,
                 add_noise: bool = False,
                 noise_std: float = 0.01):
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.add_blur = add_blur
        self.add_noise = add_noise
        self.noise_std = noise_std
    
    def __call__(self, episode: Episode) -> Episode:
        episode = deepcopy(episode)
        
        # Sample augmentation parameters once per episode
        brightness = random.uniform(*self.brightness_range)
        contrast = random.uniform(*self.contrast_range)
        
        for step in episode.steps:
            if step.observation.image:
                for camera_name, image in step.observation.image.items():
                    augmented = image.copy().astype(np.float32)
                    
                    # Brightness and contrast
                    augmented = augmented * contrast * brightness
                    
                    # Add blur
                    if self.add_blur and random.random() < 0.3:
                        kernel_size = random.choice([3, 5])
                        augmented = cv2.GaussianBlur(augmented, (kernel_size, kernel_size), 0)
                    
                    # Add noise
                    if self.add_noise:
                        noise = np.random.normal(0, self.noise_std * 255, augmented.shape)
                        augmented = augmented + noise
                    
                    # Clip values
                    augmented = np.clip(augmented, 0, 255).astype(np.uint8)
                    step.observation.image[camera_name] = augmented
        
        return episode


# ============================================================================
# Data Normalization Transforms
# ============================================================================

class NormalizeRobotState:
    """Normalize robot states using provided statistics."""
    
    def __init__(self, 
                 mean: np.ndarray, 
                 std: np.ndarray,
                 limbs: Optional[List[str]] = None):
        self.mean = mean
        self.std = std
        self.limbs = limbs
    
    def __call__(self, episode: Episode) -> Episode:
        episode = deepcopy(episode)
        
        for step in episode.steps:
            for limb_name, limb_state in step.observation.robot_state.limbs.items():
                if self.limbs is None or limb_name in self.limbs:
                    # Normalize positions
                    normalized_pos = (limb_state.joint_positions - self.mean) / (self.std + 1e-8)
                    limb_state.joint_positions = normalized_pos
                    
                    # Normalize velocities if available
                    if limb_state.joint_velocities is not None:
                        normalized_vel = (limb_state.joint_velocities - self.mean) / (self.std + 1e-8)
                        limb_state.joint_velocities = normalized_vel
        
        return episode


class ScaleActions:
    """Scale actions by a constant factor."""
    
    def __init__(self, scale_factor: float = 1.0, limbs: Optional[List[str]] = None):
        self.scale_factor = scale_factor
        self.limbs = limbs
    
    def __call__(self, episode: Episode) -> Episode:
        episode = deepcopy(episode)
        
        for step in episode.steps:
            if step.action.robot_action:
                for limb_name, action in step.action.robot_action.limbs.items():
                    if self.limbs is None or limb_name in self.limbs:
                        step.action.robot_action.limbs[limb_name] = action * self.scale_factor
        
        return episode


class ClipValues:
    """Clip robot states and actions to specified ranges."""
    
    def __init__(self, 
                 state_range: Optional[Tuple[float, float]] = None,
                 action_range: Optional[Tuple[float, float]] = None):
        self.state_range = state_range
        self.action_range = action_range
    
    def __call__(self, episode: Episode) -> Episode:
        episode = deepcopy(episode)
        
        for step in episode.steps:
            # Clip robot states
            if self.state_range:
                for limb_state in step.observation.robot_state.limbs.values():
                    limb_state.joint_positions = np.clip(
                        limb_state.joint_positions, 
                        self.state_range[0], 
                        self.state_range[1]
                    )
                    if limb_state.joint_velocities is not None:
                        limb_state.joint_velocities = np.clip(
                            limb_state.joint_velocities,
                            self.state_range[0],
                            self.state_range[1]
                        )
            
            # Clip actions
            if self.action_range and step.action.robot_action:
                for limb_name, action in step.action.robot_action.limbs.items():
                    step.action.robot_action.limbs[limb_name] = np.clip(
                        action, self.action_range[0], self.action_range[1]
                    )
        
        return episode


# ============================================================================
# Feature Engineering Transforms
# ============================================================================

class AddVelocityFeatures:
    """Compute and add velocity features to metadata."""
    
    def __call__(self, episode: Episode) -> Episode:
        episode = deepcopy(episode)
        
        for step in episode.steps:
            if step.metadata is None:
                step.metadata = {}
            
            for limb_name, limb_state in step.observation.robot_state.limbs.items():
                if limb_state.joint_velocities is not None:
                    # Velocity magnitude
                    vel_mag = np.linalg.norm(limb_state.joint_velocities)
                    step.metadata[f'{limb_name}_velocity_magnitude'] = float(vel_mag)
                    
                    # Max velocity
                    max_vel = np.max(np.abs(limb_state.joint_velocities))
                    step.metadata[f'{limb_name}_max_velocity'] = float(max_vel)
        
        return episode


class AddActionMagnitude:
    """Add action magnitude features."""
    
    def __call__(self, episode: Episode) -> Episode:
        episode = deepcopy(episode)
        
        for step in episode.steps:
            if step.action.robot_action:
                if step.metadata is None:
                    step.metadata = {}
                
                for limb_name, action in step.action.robot_action.limbs.items():
                    action_mag = np.linalg.norm(action)
                    step.metadata[f'{limb_name}_action_magnitude'] = float(action_mag)
        
        return episode


class AddEpisodeFeatures:
    """Add episode-level features to metadata."""
    
    def __call__(self, episode: Episode) -> Episode:
        episode = deepcopy(episode)
        
        if episode.meta.additional is None:
            episode.meta.additional = {}
        
        # Episode length
        episode.meta.additional['episode_length'] = len(episode.steps)
        
        # Has camera data
        has_camera = any(step.observation.image is not None for step in episode.steps)
        episode.meta.additional['has_camera_data'] = has_camera
        
        # Success rate (if step-level success available)
        step_successes = [
            step.metadata.get('success', False) 
            for step in episode.steps 
            if step.metadata
        ]
        if step_successes:
            episode.meta.additional['step_success_rate'] = np.mean(step_successes)
        
        # Average reward
        rewards = [step.reward for step in episode.steps if step.reward is not None]
        if rewards:
            episode.meta.additional['average_reward'] = np.mean(rewards)
            episode.meta.additional['total_reward'] = np.sum(rewards)
        
        return episode


# ============================================================================
# Quality Control Transforms
# ============================================================================

class ValidateEpisode:
    """Validate episode data and remove corrupted episodes."""
    
    def __init__(self, check_joint_limits: bool = True, joint_limit_range: Tuple[float, float] = (-10, 10)):
        self.check_joint_limits = check_joint_limits
        self.joint_limit_range = joint_limit_range
    
    def __call__(self, episode: Episode) -> Optional[Episode]:
        # Check for empty episode
        if len(episode.steps) == 0:
            return None
        
        # Check for NaN or infinite values
        for step in episode.steps:
            for limb_state in step.observation.robot_state.limbs.values():
                if np.any(np.isnan(limb_state.joint_positions)) or np.any(np.isinf(limb_state.joint_positions)):
                    return None
                
                if limb_state.joint_velocities is not None:
                    if np.any(np.isnan(limb_state.joint_velocities)) or np.any(np.isinf(limb_state.joint_velocities)):
                        return None
            
            # Check actions
            if step.action.robot_action:
                for action in step.action.robot_action.limbs.values():
                    if np.any(np.isnan(action)) or np.any(np.isinf(action)):
                        return None
        
        # Check joint limits
        if self.check_joint_limits:
            for step in episode.steps:
                for limb_state in step.observation.robot_state.limbs.values():
                    if (np.any(limb_state.joint_positions < self.joint_limit_range[0]) or 
                        np.any(limb_state.joint_positions > self.joint_limit_range[1])):
                        return None
        
        return episode


class RemoveStaticEpisodes:
    """Remove episodes where the robot doesn't move much."""
    
    def __init__(self, movement_threshold: float = 0.01):
        self.movement_threshold = movement_threshold
    
    def __call__(self, episode: Episode) -> Optional[Episode]:
        if len(episode.steps) < 2:
            return episode
        
        # Calculate total movement
        total_movement = 0
        
        for i in range(1, len(episode.steps)):
            curr_step = episode.steps[i]
            prev_step = episode.steps[i-1]
            
            for limb_name in curr_step.observation.robot_state.limbs:
                curr_pos = curr_step.observation.robot_state.limbs[limb_name].joint_positions
                prev_pos = prev_step.observation.robot_state.limbs[limb_name].joint_positions
                
                movement = np.linalg.norm(curr_pos - prev_pos)
                total_movement += movement
        
        if total_movement < self.movement_threshold:
            return None
        
        return episode


# ============================================================================
# Transform Composition Utilities
# ============================================================================

class Compose:
    """Compose multiple transforms together."""
    
    def __init__(self, transforms: List[Callable[[Episode], Optional[Episode]]]):
        self.transforms = transforms
    
    def __call__(self, episode: Episode) -> Optional[Episode]:
        for transform in self.transforms:
            episode = transform(episode)
            if episode is None:  # Episode was filtered out
                return None
        return episode


class RandomApply:
    """Randomly apply a transform with given probability."""
    
    def __init__(self, transform: Callable[[Episode], Episode], probability: float = 0.5):
        self.transform = transform
        self.probability = probability
    
    def __call__(self, episode: Episode) -> Episode:
        if random.random() < self.probability:
            return self.transform(episode)
        return episode


# ============================================================================
# Common Transform Combinations
# ============================================================================

def create_training_transforms(noise_std: float = 0.01, augment_images: bool = True) -> Compose:
    """Create a standard set of training transforms."""
    transforms = [
        ValidateEpisode(),
        FilterBySuccess(success_only=True),
        TruncateEpisode(max_steps=200),
        AddNoiseToRobotState(position_noise_std=noise_std),
        RandomApply(AddNoiseToActions(noise_std=noise_std), probability=0.5),
        AddVelocityFeatures(),
        AddActionMagnitude(),
        AddEpisodeFeatures()
    ]
    
    if augment_images:
        transforms.append(RandomApply(ImageAugmentation(), probability=0.3))
    
    return Compose(transforms)


def create_validation_transforms() -> Compose:
    """Create transforms for validation (no augmentation)."""
    return Compose([
        ValidateEpisode(),
        TruncateEpisode(max_steps=200),
        AddVelocityFeatures(),
        AddActionMagnitude(), 
        AddEpisodeFeatures()
    ])


def create_filtering_transforms(success_only: bool = True, min_steps: int = 10) -> Compose:
    """Create transforms focused on data filtering."""
    return Compose([
        ValidateEpisode(),
        FilterBySuccess(success_only=success_only),
        FilterByLength(min_steps=min_steps),
        RemoveStaticEpisodes()
    ])