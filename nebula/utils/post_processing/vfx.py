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
    _data['degree'] = ppm_data[2] if len(ppm_data) > 2 else 1.0  # e.g., '1'
    return _data


def _add_possion_gaussian_noise(cam_data, lam):
    """
    rgb+pg_noise+10
    Add Poisson-Gaussian noise to camera data. inplace"""
    # cam_data.shape = torch.Size([1, 512, 512, 3])
    tgt = cam_data.float()
    lam_tensor = torch.clamp(tgt / 255.0 * lam, min=0.0)   # λ control here
    poisson = torch.poisson(lam_tensor)
    gaussian = torch.normal(mean=0.0, std=lam, size=tgt.shape, device=tgt.device)
    tgt += poisson + gaussian
    tgt.clamp_(0.0, 255.0)
    cam_data.copy_(tgt.to(cam_data.dtype))
    return cam_data

def _add_rolling_shutter_effect(cam_data, ratio, direction="right", curve="sqrt"):
    """
    rgb+rolling_shutter+0.05
    In-place rolling-shutter skew using a width ratio.
    - ratio: 0..1 fraction of width (max per-row shift)
    - direction: "right" or "left"
    - curve: "linear" | "sqrt" | "square" (how offset grows down rows)
    """
    assert cam_data.ndim == 4, "expected [B, H, W, C]"
    ratio = float(ratio)
    B, H, W, C = cam_data.shape
    if H == 0 or W == 0 or ratio == 0:
        return cam_data

    # Max shift in pixels from ratio, clamped to < W
    ratio = float(max(0.0, min(1.0, ratio)))
    max_shift = min(int(round(ratio * (W - 1))), W - 1)
    if max_shift == 0:
        return cam_data

    # Per-row offsets
    t = torch.linspace(0.0, 1.0, steps=H, device=cam_data.device)
    if curve == "sqrt":
        t = torch.sqrt(t)
    elif curve == "square":
        t = t * t
    elif curve != "linear":
        raise ValueError("curve must be 'linear', 'sqrt', or 'square'")
    offsets = (t * max_shift).round().to(torch.long)  # [H]

    # Build clamped gather indices (edge replication, no wrap)
    sign = 1 if direction == "right" else -1
    base_cols = torch.arange(W, device=cam_data.device).view(1, 1, W, 1)   # [1,1,W,1]
    per_row = (offsets.view(1, H, 1, 1) * sign)                             # [1,H,1,1]
    idx_cols = (base_cols - per_row).clamp_(0, W - 1).expand(B, H, W, C).long()

    # Apply skew
    cam_data.copy_(torch.gather(cam_data, dim=2, index=idx_cols))
    return cam_data


post_process_func_hash = {"pg_noise":_add_possion_gaussian_noise,
                         "rolling_shutter": _add_rolling_shutter_effect
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

                    obs['sensor_data'][cam_name][ppm_data['mode']] = pp_function(cam_data,ppm_data['degree'])

            return obs, reward, terminated, truncated, info
        
        return wrapper
    
    return _decorator


