import os
import importlib.util
from pathlib import Path
from typing import Dict, Any, Union, Optional, List, Tuple


class Embodiment:
    """Load and parse robot configuration from Python config files."""
    
    def __init__(self, config_path: Union[str, Path], robot_name: str):
        self.config_path = Path(config_path)
        self.robot_name = robot_name
        self.config = self._load_config()
        self.robot_type = self.config.get('robot_type', 'single_arm')
        self.limbs = self._parse_limbs()
        self.modality = self.config.get('modality', {})
        self.views = self.config.get('views', {})
        self.is_dual_arm = self.robot_type == 'dual_arm'
    
    def _load_config(self) -> Dict:
        """Load robot configuration from Python file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Robot config file not found: {self.config_path}")
        
        # Load the Python module
        spec = importlib.util.spec_from_file_location("embodiment_config", self.config_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load config from: {self.config_path}")
        
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        
        # Get the ROBOT_CONFIGS dictionary
        if not hasattr(config_module, 'ROBOT_CONFIGS'):
            raise AttributeError(f"Config file must contain a 'ROBOT_CONFIGS' dictionary")
        
        robot_configs = getattr(config_module, 'ROBOT_CONFIGS')
        
        # Get the specific robot configuration
        if self.robot_name not in robot_configs:
            available_robots = list(robot_configs.keys())
            raise KeyError(f"Robot '{self.robot_name}' not found in config. Available robots: {available_robots}")
        
        return robot_configs[self.robot_name]
        
    def _parse_limbs(self) -> Dict[str, Dict]:
        """Parse limb configuration dynamically."""
        limbs = {}
        if 'limbs' in self.config:
            for limb_name, limb_config in self.config['limbs'].items():
                # Simply copy the limb config as is
                limbs[limb_name] = limb_config.copy()
        return limbs
    
    def get_limb_info(self, limb_name: str) -> Optional[Dict]:
        """Get information for a specific limb."""
        # Special handling for dual-arm robots with left/right prefix
        if self.is_dual_arm and limb_name.startswith(('left_', 'right_')):
            # Extract the base limb name (e.g., "left_arm" -> "arm")
            base_limb = limb_name[len('left_'):] if limb_name.startswith('left_') else limb_name[len('right_'):]
            return self.limbs.get(base_limb)
        return self.limbs.get(limb_name)
    
    def get_all_limb_names(self) -> List[str]:
        """Get names of all limbs."""
        if not self.is_dual_arm:
            # Standard behavior for single arm robots
            return list(self.limbs.keys())
        else:
            # For dual arm robots, return left_ and right_ prefixed limbs
            all_limbs = []
            for limb_name in self.limbs.keys():
                all_limbs.append(f"left_{limb_name}")
                all_limbs.append(f"right_{limb_name}")
            return all_limbs
    
    def get_action_dim(self, limb_name: str) -> int:
        """Get action dimension for a specific limb."""
        limb_info = self.get_limb_info(limb_name)
        if not limb_info:
            return 0
        
        dof = limb_info.get('dof', 0)
        
        # No change needed for single arm robots
        if not self.is_dual_arm:
            return dof
        
        # For dual arm robots, we still return the same DOF for left/right limbs
        return dof
    
    def get_joint_names(self, limb_name: str) -> List[str]:
        """Get joint names for a specific limb."""
        limb_info = self.get_limb_info(limb_name)
        if not limb_info:
            return []
        
        # For single arm robots or non-prefixed limb names
        if not self.is_dual_arm or not (limb_name.startswith('left_') or limb_name.startswith('right_')):
            return limb_info.get('joint_names', [])
        
        # For dual arm robots, use the appropriate joint names based on prefix
        if limb_name.startswith('left_'):
            return limb_info.get('left_joint_names', [])
        elif limb_name.startswith('right_'):
            return limb_info.get('right_joint_names', [])
        
        return []
    
    def get_total_action_dim(self) -> int:
        """Get total action dimension across all limbs."""
        if not self.is_dual_arm:
            # Standard behavior for single arm robots
            return sum(limb.get('dof', 0) for limb in self.limbs.values())
        else:
            # For dual arm robots, each limb appears twice (left and right)
            return 2 * sum(limb.get('dof', 0) for limb in self.limbs.values())
    
    def is_dual_arm_robot(self) -> bool:
        """Check if this is a dual-arm robot."""
        return self.is_dual_arm
    
    # ====== Modality Configuration Methods ======
    
    def get_state_config(self, limb_name: str) -> Optional[Dict]:
        """Get state configuration for a specific limb."""
        state_config = self.modality.get('state', {})
        return state_config.get(limb_name)
    
    def get_action_config(self, limb_name: str) -> Optional[Dict]:
        """Get action configuration for a specific limb."""
        action_config = self.modality.get('action', {})
        return action_config.get(limb_name)
    
    def get_state_slice(self, limb_name: str) -> Optional[Tuple[int, int]]:
        """Get state slice indices (start, end) for a specific limb."""
        state_config = self.get_state_config(limb_name)
        if state_config and 'start' in state_config and 'end' in state_config:
            return (state_config['start'], state_config['end'])
        return None
    
    def get_action_slice(self, limb_name: str) -> Optional[Tuple[int, int]]:
        """Get action slice indices (start, end) for a specific limb."""
        action_config = self.get_action_config(limb_name)
        if action_config and 'start' in action_config and 'end' in action_config:
            return (action_config['start'], action_config['end'])
        return None
    
    def get_state_dataset_key(self, limb_name: str) -> Optional[str]:
        """Get dataset key for state data of a specific limb."""
        state_config = self.get_state_config(limb_name)
        if state_config:
            return state_config.get('dataset_key')
        return None
    
    def get_action_dataset_key(self, limb_name: str) -> Optional[str]:
        """Get dataset key for action data of a specific limb."""
        action_config = self.get_action_config(limb_name)
        if action_config:
            return action_config.get('dataset_key')
        return None
    
    def is_state_absolute(self, limb_name: str) -> bool:
        """Check if state data for a limb is absolute (vs relative)."""
        state_config = self.get_state_config(limb_name)
        if state_config:
            return state_config.get('absolute', True)
        return True
    
    def is_action_absolute(self, limb_name: str) -> bool:
        """Check if action data for a limb is absolute (vs relative)."""
        action_config = self.get_action_config(limb_name)
        if action_config:
            return action_config.get('absolute', False)
        return False
    
    def get_all_state_limbs(self) -> List[str]:
        """Get all limb names that have state configuration."""
        return list(self.modality.get('state', {}).keys())
    
    def get_all_action_limbs(self) -> List[str]:
        """Get all limb names that have action configuration."""
        return list(self.modality.get('action', {}).keys())
    
    # ====== View Configuration Methods ======
    
    def get_view_config(self, view_name: str) -> Optional[Dict]:
        """Get configuration for a specific view/camera."""
        return self.views.get(view_name)
    
    def get_view_dataset_key(self, view_name: str) -> Optional[str]:
        """Get dataset key for a specific view/camera."""
        view_config = self.get_view_config(view_name)
        if view_config:
            return view_config.get('dataset_key')
        return None
    
    def get_all_view_names(self) -> List[str]:
        """Get names of all available views/cameras."""
        return list(self.views.keys())
    
    def get_all_view_dataset_keys(self) -> Dict[str, str]:
        """Get mapping of view names to their dataset keys."""
        return {view_name: config.get('dataset_key', '') 
                for view_name, config in self.views.items() 
                if 'dataset_key' in config}
    
    # ====== Utility Methods ======
    
    def get_full_state_slice_mapping(self) -> Dict[str, Tuple[int, int]]:
        """Get mapping of all limbs to their state slice indices."""
        mapping = {}
        for limb_name in self.get_all_state_limbs():
            slice_info = self.get_state_slice(limb_name)
            if slice_info:
                mapping[limb_name] = slice_info
        return mapping
    
    def get_full_action_slice_mapping(self) -> Dict[str, Tuple[int, int]]:
        """Get mapping of all limbs to their action slice indices."""
        mapping = {}
        for limb_name in self.get_all_action_limbs():
            slice_info = self.get_action_slice(limb_name)
            if slice_info:
                mapping[limb_name] = slice_info
        return mapping
    
    def get_total_state_dim(self) -> int:
        """Get total state dimension across all limbs based on modality config."""
        max_end = 0
        for limb_name in self.get_all_state_limbs():
            slice_info = self.get_state_slice(limb_name)
            if slice_info:
                _, end = slice_info
                max_end = max(max_end, end)
        return max_end
    
    def get_total_action_dim_from_modality(self) -> int:
        """Get total action dimension across all limbs based on modality config."""
        max_end = 0
        for limb_name in self.get_all_action_limbs():
            slice_info = self.get_action_slice(limb_name)
            if slice_info:
                _, end = slice_info
                max_end = max(max_end, end)
        return max_end
    
    @classmethod
    def list_available_robots(cls, config_path: Union[str, Path]) -> List[str]:
        """List all available robot configurations in the config file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Robot config file not found: {config_path}")
        
        # Load the Python module
        spec = importlib.util.spec_from_file_location("robot_config", config_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load config from: {config_path}")
        
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        
        # Get the ROBOT_CONFIGS dictionary
        if not hasattr(config_module, 'ROBOT_CONFIGS'):
            raise AttributeError(f"Config file must contain a 'ROBOT_CONFIGS' dictionary")
        
        robot_configs = getattr(config_module, 'ROBOT_CONFIGS')
        return list(robot_configs.keys())