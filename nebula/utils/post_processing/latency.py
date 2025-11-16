from functools import wraps
from collections import defaultdict,deque
import random
import numpy as np

G_ACTION_BUFFER_SIZE = 128
G_ACTION_BUFFER_HASH = defaultdict(lambda: deque(maxlen=G_ACTION_BUFFER_SIZE))
G_FIRST_ACTION_HASH = {}  # pid -> first action to prefill buffer
G_PERVIOUS_ACTION_HASH = {}  # pid -> previous action
def _post_process_mode_parser(mode_str):
    """
    Parse post process mode string into a set of processing flags.
    """
    ppm_data = mode_str.split('+')
    _data = {}
    _data['post_processes'] = str(ppm_data[0]).lower()  # e.g., ['pg_noise', '1']
    _data['degree'] = ppm_data[1] if len(ppm_data) > 1 else 1.0  # e.g., '1'
    return _data

def _add_latency(pid, action, latency_steps: float):
    """
    latency+2
    Add action latency by buffering actions.
    - latency_steps: number of steps to delay action
    """
    latency_steps = int(max(0, latency_steps))
    if latency_steps == 0:
        return action
    
    action_buffer = G_ACTION_BUFFER_HASH[pid]
    
    if pid not in G_FIRST_ACTION_HASH:
        G_FIRST_ACTION_HASH[pid] = action
  
    # action <class 'numpy.ndarray'>
    
    if len(action_buffer) < latency_steps:
        action_buffer.append(action)

        first_action = G_FIRST_ACTION_HASH[pid]
        return first_action
    else:
        action_buffer.append(action)
        delayed_action = action_buffer.popleft()
    return delayed_action

def _add_packet_drop(pid, action, drop_rate: float):
    """
    packet_drop+0.1
    Simulate packet drop by randomly dropping actions (to Previous action).
    - drop_rate: probability of dropping an action
    """
    if pid not in G_PERVIOUS_ACTION_HASH:
        G_PERVIOUS_ACTION_HASH[pid] = action

    drop_rate = max(0.0, min(1.0, drop_rate))

    if random.random() < drop_rate:
        # Drop the action, return previous action
        return G_PERVIOUS_ACTION_HASH[pid]
        
    G_PERVIOUS_ACTION_HASH[pid] = action

    return action

post_process_func_hash = {"latency":_add_latency,
                         "packet_drop": _add_packet_drop
                         }

def latency_filter(fn=None):
    def _decorator(step_fn):
        """Post-process only the action part of `step`."""
        @wraps(step_fn)
        def wrapper(self, action):
            # get process ID
            pid = id(self)
            # parse post process method
            if isinstance(self.post_processing_method, str):
                ppm_list = [self.post_processing_method]
            else:
                ppm_list = self.post_processing_method
            for ppm in ppm_list:
                ppm_data = _post_process_mode_parser(ppm)
                p_process_name = ppm_data['post_processes']
                degree = ppm_data['degree']
                if p_process_name in post_process_func_hash:
                    action = post_process_func_hash[p_process_name](pid, action, float(degree))
                else:
                    raise ValueError(f"Unsupported post process method: {p_process_name}")
            
            obs, reward, terminated, truncated, info = step_fn(self, action)
            
            return obs, reward, terminated, truncated, info
        
        return wrapper
    
    return _decorator

