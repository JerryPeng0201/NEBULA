from functools import wraps
from nebula.core.simulation.engine import BaseEnv
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def _post_process_mode_parser(mode_str):
    """
    Parse post process mode string into a set of processing flags.
    """
    ppm_data = mode_str.split('+')
    _data = {}
    _data['mode'] = str(ppm_data[0]).lower()  # e.g., 'rgb', 'depth', 'segmentation'
    _data['post_processes'] = str(ppm_data[1]).lower()  # e.g., ['pg_noise', '1']
    _data['degree'] = ppm_data[2] if len(ppm_data) > 2 else 1.0  # e.g., '1'
    return _data


def _add_possion_gaussian_noise(cam_data, lam):
    """
    rgb+pg_noise+10
    Add Poisson-Gaussian noise to camera data. inplace
    - lam: noise level
    """
    # cam_data.shape = torch.Size([1, 512, 512, 3])
    tgt = cam_data.float()
    lam_tensor = torch.clamp(tgt / 255.0 * lam, min=0.0)   # λ control here
    poisson = torch.poisson(lam_tensor)
    gaussian = torch.normal(mean=0.0, std=lam, size=tgt.shape, device=tgt.device)
    tgt += poisson + gaussian
    tgt.clamp_(0.0, 255.0)
    cam_data.copy_(tgt.to(cam_data.dtype))
    # show image for debug
    # plt.imshow(cam_data[0].cpu().numpy().astype(np.uint8))
    # plt.show()
    return cam_data

def _add_rolling_shutter_effect(cam_data, ratio, direction="right", curve="sqrt"):
    """
    rgb+rolling_shutter+0.05
    In-place rolling-shutter skew using a width ratio.
    - ratio: 0..1 fraction of width (max per-row shift)
    - direction: "right" or "left"
    - curve: "linear" | "sqrt" | "square" (how offset grows down rows)
    """
    assert cam_data.ndim == 4, "expected [1, H, W, C]"
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
    # show image for debug
    # plt.imshow(cam_data[0].cpu().numpy().astype(np.uint8))
    # plt.show()
    return cam_data

def _add_light_flicker_effect(cam_data, frequency, amplitude=0.1):
    """
    rgb+light_flicker+50
    In-place light flicker effect using a sine wave.
    - frequency: frequency of flicker (Hz)
    - amplitude: amplitude of flicker (0..1)
    """
    assert cam_data.ndim == 4, "expected [1, H, W, C]"
    frequency = float(frequency)
    amplitude = float(amplitude)
    B, H, W, C = cam_data.shape
    if H == 0 or W == 0 or frequency == 0 or amplitude == 0:
        return cam_data

    # Create flicker mask
    t = torch.linspace(0.0, 2 * np.pi * frequency, steps=H, device=cam_data.device)
    flicker = (1.0 + amplitude * torch.sin(t)).view(1, H, 1, 1)  # [1,H,1,1]

    # Apply flicker
    tgt = cam_data.float() * flicker
    tgt.clamp_(0.0, 255.0)
    cam_data.copy_(tgt.to(cam_data.dtype))
    # show image for debug
    # plt.imshow(cam_data[0].cpu().numpy().astype(np.uint8))
    # plt.show()
    return cam_data

def _add_resolution_degradation(cam_data, scale_factor):
    """
    rgb+resolution_degradation+4 (checkboard pattern show at 4)
    In-place resolution degradation using downsampling
    - scale_factor: 2,4,8 fraction to downscale resolution
    """
    assert cam_data.ndim == 4, "expected [1, H, W, C]"
    B, H, W, C = cam_data.shape
    if H == 0 or W == 0:
        return cam_data

    scale = float(scale_factor)
    if scale <= 0.0 or scale == 1.0:
        return cam_data

    # Compute downscale ratio in (0, 1]
    down_ratio = 1.0 / scale if scale > 1.0 else scale
    new_H = max(1, int(round(H * float(down_ratio))))
    new_W = max(1, int(round(W * float(down_ratio))))

    x = cam_data.permute(0, 3, 1, 2).contiguous().to(dtype=torch.float32)  # NCHW
    x_down = F.interpolate(x, size=(new_H, new_W), mode="area")
    x_up = F.interpolate(x_down, size=(H, W), mode="bilinear", align_corners=False)

    # Preserve dtype and range
    if not torch.is_floating_point(cam_data):
        x_up = x_up.clamp_(0.0, 255.0).to(dtype=cam_data.dtype)
    else:
        x_up = x_up.to(dtype=cam_data.dtype)
    cam_data.copy_(x_up.permute(0, 2, 3, 1).contiguous())
    # show image for debug
    # plt.imshow(cam_data[0].cpu().numpy().astype(np.uint8))
    # plt.show()
    return cam_data

def _add_frame_drop_effect(cam_data, drop_rate):
    """
    rgb+frame_drop+0.1
    In-place frame drop to 0 effect using a local step drop rate.
    - drop_rate: 0..1 fraction of frames to drop
    """
    assert cam_data.ndim == 4, "expected [1, H, W, C]"
    drop_rate = float(drop_rate)
    B, H, W, C = cam_data.shape
    if H == 0 or W == 0 or drop_rate <= 0.0:
        return cam_data 
    if np.random.rand() < drop_rate:
        cam_data.zero_()
    #show image for debug
    # plt.imshow(cam_data[0].cpu().numpy().astype(np.uint8))
    # plt.show()
    return cam_data

def _add_color_shift_effect(cam_data, rgb_value):
    """
    rgb+color_shift+30,20,10
    In-place color shift effect using rgb value shift.
    - rgb_value: r,g,b value shift
    """
    assert cam_data.ndim == 4, "expected [1, H, W, C]"
    B, H, W, C = cam_data.shape
    if H == 0 or W == 0:
        return cam_data
    rgb_shift = [int(v) for v in rgb_value.split(',')]
    shift = torch.as_tensor(rgb_shift, device=cam_data.device, dtype=torch.float32).view(1, 1, 1, 3)

    if cam_data.is_floating_point():
        cam_data.add_(shift)
        cam_data.clamp_(0.0, 255.0)
        return cam_data

    # Non-float (e.g., uint8): compute in float, then write back in place
    tmp = cam_data.to(torch.float32)
    tmp.add_(shift)
    tmp.clamp_(0.0, 255.0)
    cam_data.copy_(tmp.to(dtype=cam_data.dtype))
    # show image for debug
    plt.imshow(cam_data[0].cpu().numpy().astype(np.uint8))
    plt.show()
    return cam_data
    

post_process_func_hash = {"pg_noise":_add_possion_gaussian_noise,
                         "rolling_shutter": _add_rolling_shutter_effect,
                         "light_flicker": _add_light_flicker_effect,
                         "resolution_degradation": _add_resolution_degradation,
                         "frame_drop": _add_frame_drop_effect,
                         "color_shift": _add_color_shift_effect
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

                for cam_name in obs['sensor_data'].keys():
               
                    cam_data = obs['sensor_data'][cam_name][ppm_data['mode']]
                    
                    pp_function = post_process_func_hash.get(ppm_data['post_processes'])

                    obs['sensor_data'][cam_name][ppm_data['mode']] = pp_function(cam_data,ppm_data['degree'])
            return obs, reward, terminated, truncated, info
        
        return wrapper
    
    return _decorator


