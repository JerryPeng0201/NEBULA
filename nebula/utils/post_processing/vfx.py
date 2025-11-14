from functools import wraps
from nebula.core.simulation.engine import BaseEnv
import numpy as np
import torch
import matplotlib.pyplot as plt

def _post_process_mode_parser(mode_str):
    """Parse post process mode string into a set of processing flags."""
    ppm_data = mode_str.split('+')
    _data = {}
    _data['mode'] = str(ppm_data[0]).lower()  # e.g., 'rgb', 'depth', 'segmentation'
    _data['post_processes'] = str(ppm_data[1]).lower()  # e.g., ['pg_noise', '1']
    _data['degree'] = ppm_data[2] if len(ppm_data) > 2 else 1  # e.g., '1'
    return _data


def _add_possion_gaussian_noise(cam_data, lam):
    """Add Poisson-Gaussian noise to camera data. inplace"""
    # cam_data.shape = torch.Size([1, 512, 512, 3])
    tgt = cam_data.float()
    lam_tensor = torch.clamp(tgt / 255.0 * lam, min=0.0)   # λ control here
    poisson = torch.poisson(lam_tensor)
    gaussian = torch.normal(mean=0.0, std=lam, size=tgt.shape, device=tgt.device)
    tgt += poisson + gaussian
    tgt.clamp_(0.0, 255.0)
    cam_data.copy_(tgt.to(cam_data.dtype))
    return cam_data



post_process_func_hash = {"pg_noise":_add_possion_gaussian_noise

}

def obs_filter(fn=None,*, post_process_mode=None):
     
    def _decorator(step_fn):
        """Post-process only the observation part of `step`."""
        @wraps(step_fn)
        def wrapper(self, action):
            # assert fn heritages from BaseEnv.step
            assert step_fn.__name__ == "step", "obs_filter can only decorate step() implementations"
            assert isinstance(self, BaseEnv), "obs_filter must wrap BaseEnv.step overrides"
        
            obs, reward, terminated, truncated, info = step_fn(self, action)
            if post_process_mode is not None:
                
                ppm_data = _post_process_mode_parser(post_process_mode)
                print("Post process mode:", ppm_data)
                print(obs['sensor_data'].keys())
                for cam_name in obs['sensor_data'].keys():
               
                    cam_data = obs['sensor_data'][cam_name][ppm_data['mode']]
                    
                    pp_function = post_process_func_hash.get(ppm_data['post_processes'])

                    obs['sensor_data'][cam_name][ppm_data['mode']] = pp_function(cam_data,int(ppm_data['degree']))

            return obs, reward, terminated, truncated, info
        
        return wrapper
    
    return _decorator


