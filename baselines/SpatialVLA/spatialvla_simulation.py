import sys
sys.path.append('./SpatialVLA')
sys.path.append('../..')

import gymnasium as gym
import numpy as np
import nebula.benchmarks
import argparse
import yaml
import torch
import time
import json
import os
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import imageio
from PIL import Image
from transformers import AutoModel, AutoProcessor
import psutil

TASK_DESCRIPTIONS = {
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

def get_gpu_memory_usage():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024**2
        gpu_memory_max = torch.cuda.max_memory_allocated() / 1024**2
        gpu_memory_reserved = torch.cuda.memory_reserved() / 1024**2
        return {
            'allocated': gpu_memory,
            'max_allocated': gpu_memory_max,
            'reserved': gpu_memory_reserved
        }
    return None

def get_cpu_memory_usage():
    """Get current CPU memory usage in MB."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return {
        'rss': memory_info.rss / 1024**2,
        'vms': memory_info.vms / 1024**2,
        'percent': process.memory_percent()
    }

def count_parameters(model):
    """Count the total number of parameters in a model."""
    if hasattr(model, 'model') and hasattr(model.model, 'parameters'):
        total_params = sum(p.numel() for p in model.model.parameters())
        trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    elif hasattr(model, 'parameters'):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        # Try to access nested models like transformers, diffusion networks
        total_params = 0
        trainable_params = 0
        for attr_name in dir(model):
            attr = getattr(model, attr_name)
            if hasattr(attr, 'parameters'):
                total_params += sum(p.numel() for p in attr.parameters())
                trainable_params += sum(p.numel() for p in attr.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }

def calculate_stability_score(action_history):
    if len(action_history) < 2:
        return 1.0
    action_history = np.array(action_history)
    action_changes = np.diff(action_history, axis=0)
    change_magnitudes = np.linalg.norm(action_changes, axis=1)
    mean_change = np.mean(change_magnitudes)
    return float(np.exp(-mean_change))

def create_spatialvla_policy(config):
    """Create SpatialVLA policy from config."""
    processor = AutoProcessor.from_pretrained(
        config['model']['model_path'], 
        trust_remote_code=True
    )
    model = AutoModel.from_pretrained(
        config['model']['model_path'], 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16
    ).eval().cuda()
    
    return model, processor

def nebula_to_spatialvla_obs(nebula_obs, task_description, info=None):
    """Convert nebula observation to SpatialVLA format."""
    # Extract base camera image
    base_camera = nebula_obs['sensor_data']['base_camera']['rgb'].detach().cpu().numpy()
    
    # Handle shape - squeeze extra dimensions
    while base_camera.ndim > 3:
        base_camera = base_camera.squeeze(0)
    
    # Convert to uint8 if needed
    if base_camera.dtype != np.uint8:
        if base_camera.max() <= 1.0:
            base_camera = (base_camera * 255).astype(np.uint8)
        else:
            base_camera = base_camera.astype(np.uint8)
    
    # Convert to PIL Image
    image = Image.fromarray(base_camera)
    
    # Update task description from info if available
    if info is not None and 'task_instruction' in info:
        task_description = info['task_instruction']
    
    return image, task_description

def spatialvla_to_nebula_action(spatialvla_actions):
    """Convert SpatialVLA action to nebula format."""
    # Extract actions from dict
    if isinstance(spatialvla_actions, dict):
        if 'actions' in spatialvla_actions:
            actions = spatialvla_actions['actions']
        else:
            raise ValueError(f"Expected 'actions' key in dict, got keys: {spatialvla_actions.keys()}")
    else:
        actions = spatialvla_actions
    
    # Convert to numpy if tensor
    if torch.is_tensor(actions):
        actions = actions.detach().cpu().numpy()
    else:
        actions = np.array(actions)
    
    # Handle shape - extract first action if batched
    if actions.ndim == 2 and actions.shape[0] == 1:
        actions = actions[0]
    
    # SpatialVLA outputs 7D EEF action [dx, dy, dz, drx, dry, drz, gripper]
    # Convert to 8D joint action for Nebula
    eef_action = actions[:6]  # translation + rotation
    gripper_action = actions[6]  # gripper
    
    # Create 8D joint action
    joint_action = np.zeros(8)
    joint_action[:6] = eef_action * 0.1  # Scale EEF movements for joint space
    joint_action[6] = gripper_action
    joint_action[7] = gripper_action
    
    # Return as list of actions (single action for now)
    return [joint_action]

class SpatialVLAPolicy:
    """Wrapper class for SpatialVLA policy - mimics Gr00tPolicy interface."""
    
    def __init__(self, model, processor, config):
        self.model = model
        self.processor = processor
        self.unnorm_key = config['model'].get('unnorm_key', 'nebula')
        
    def get_action(self, obs_dict):
        """Get action from SpatialVLA model - matches Gr00tPolicy.get_action signature."""
        # Extract image and task description
        image = obs_dict['image']
        task_description = obs_dict['task_description']
        
        # Process inputs
        inputs = self.processor(
            images=[image], 
            text=task_description, 
            unnorm_key=self.unnorm_key,
            return_tensors="pt"
        )
        
        # Move inputs to GPU
        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].cuda()
        
        # Get action prediction
        with torch.no_grad():
            generation_outputs = self.model.predict_action(inputs)
            actions = self.processor.decode_actions(generation_outputs, unnorm_key=self.unnorm_key)
        
        return actions

def run_episode(env, policy, env_id, config, episode_idx, save_video=False, video_dir=None):
    obs, _ = env.reset(seed=episode_idx + config['experiment']['base_seed'])
    action_history = []
    episode_inference_times = []
    video_frames = []
    global_steps = 0
    task_description_record = ""
    done = False
    episode_gpu_memory_peak = 0
    episode_cpu_memory_peak = 0

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    if hasattr(env, 'evaluate'):
        info = env.evaluate()
    else:
        info = {}
    
    while global_steps < config['experiment']['max_episode_steps'] and not done:
        
        # Get task description
        if hasattr(env.unwrapped, 'get_task_instruction'):
            task_description = env.unwrapped.get_task_instruction()
        else:
            task_description = TASK_DESCRIPTIONS.get(env_id, "Complete the task")

        # Record task description only if it changes
        if not task_description_record.endswith(task_description):
            if task_description_record != "":
                task_description_record += " THEN "
            task_description_record += task_description
        
        # Convert observation to SpatialVLA format
        image, task_desc = nebula_to_spatialvla_obs(obs, task_description, info)
        
        # Prepare observation dict for policy
        policy_obs = {
            'image': image,
            'task_description': task_desc
        }
        
        # Get action from policy
        start_time = time.perf_counter()
        spatialvla_action = policy.get_action(policy_obs)
        inference_time = time.perf_counter() - start_time
        episode_inference_times.append(inference_time)

        # Monitor memory after inference
        post_gpu = get_gpu_memory_usage()
        post_cpu = get_cpu_memory_usage()

        if post_gpu:
            episode_gpu_memory_peak = max(episode_gpu_memory_peak, post_gpu['allocated'])
        if post_cpu:
            episode_cpu_memory_peak = max(episode_cpu_memory_peak, post_cpu['rss'])
        
        # Convert to nebula actions
        nebula_actions = spatialvla_to_nebula_action(spatialvla_action)
        
        # Execute actions
        for action in nebula_actions:
            if global_steps >= config['experiment']['max_episode_steps'] or done:
                break
            obs, reward, terminated, truncated, info = env.step(action)
            action_history.append(action)
            
            if save_video:
                frame = env.render().squeeze(0).detach().cpu().numpy()
                video_frames.append(frame)
            
            global_steps += 1
            
            if terminated or truncated:
                if info.get('success', False):
                    done = True
                    break
    
    # Save video if enabled
    if save_video and video_dir and len(video_frames) > 0:
        video_path = video_dir / f"{env_id}_episode_{episode_idx}.mp4"
        imageio.mimsave(video_path, video_frames, fps=20)
    
    return {
        'task_instruction': task_description_record,
        'success': info.get('success', False),
        'steps': global_steps,
        'avg_inference_time': float(np.mean(episode_inference_times)),
        'stability_score': calculate_stability_score(action_history),
        'gpu_memory_peak': episode_gpu_memory_peak,
        'cpu_memory_peak': episode_cpu_memory_peak,
    }

def evaluate_task(env_id, policy, config):
    """Evaluate policy on a specific task."""
    env = gym.make(
        env_id,
        obs_mode=config['environment']['obs_mode'],
        control_mode=config['environment']['control_mode'],
        render_mode=config['environment']['render_mode'],
        reward_mode=config['environment']['reward_mode'],
        sensor_configs=dict(shader_pack=config['environment']['shader']),
        human_render_camera_configs=dict(shader_pack=config['environment']['shader']),
        viewer_camera_configs=dict(shader_pack=config['environment']['shader']),
        sim_backend=config['environment']['sim_backend'],
        reconfiguration_freq=1
    )
    
    save_video = config.get('video', {}).get('save_video', False)
    video_dir = None
    if save_video:
        video_dir = Path(config.get('video', {}).get('save_dir', './videos')) / env_id
        video_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    for episode in tqdm(range(config['experiment']['num_traj']), 
                       desc=f"Evaluating {env_id}", 
                       leave=False):
        episode_result = run_episode(env, policy, env_id, config, episode, 
                                    save_video=save_video, video_dir=video_dir)
        results.append(episode_result)
    
    env.close()
    
    successes = [r['success'] for r in results]
    inference_times = [r['avg_inference_time'] for r in results]
    stability_scores = [r['stability_score'] for r in results]
    
    return {
        'task': env_id,
        'success_rate': float(np.mean(successes) * 100),
        'avg_inference_frequency_hz': float(1.0 / np.mean(inference_times)),
        'avg_latency_ms': float(np.mean(inference_times) * 1000),
        'avg_stability_score': float(np.mean(stability_scores)),
        'avg_gpu_memory_peak_mb': float(np.mean([r['gpu_memory_peak'] for r in results])),
        'avg_cpu_memory_peak_mb': float(np.mean([r['cpu_memory_peak'] for r in results])),
        'num_trajectories': config['experiment']['num_traj'],
        'episodes': results
    }

def make_json_serializable(obj):
    """Recursively convert objects to JSON-serializable types."""
    if isinstance(obj, (torch.Tensor, np.ndarray)):
        return obj.tolist() if hasattr(obj, 'tolist') else str(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    else:
        return obj
    
def generate_conclusion(config, test_type):
    """Generate high-level summary of all test results."""
    output_dir = Path(config['experiment']['save_dir'])
    session_id = config.get('session_id')
    results_file = output_dir / f'results_{test_type}_{session_id}.json'
    
    if not results_file.exists():
        print(f"No results file found: {results_file}")
        return
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    summary = {
        'session_id': session_id,
        'test_type': test_type,
        'timestamp': datetime.now().isoformat(),
        'model_name': config.get('figure', {}).get('model_name', 'diffusion_policy'),
        'overall_metrics': {},
        'category_breakdown': {},
        'difficulty_breakdown': {}
    }
    
    all_success_rates = []
    all_inference_freqs = []
    all_latencies = []
    all_stabilities = []
    all_gpu_peaks = []
    all_cpu_peaks = []
    
    category_results = {}
    difficulty_results = {'Easy': [], 'Medium': [], 'Hard': []}
    
    for task_result in data['results']:
        task_name = task_result['task']
        success_rate = task_result['success_rate']
        
        all_success_rates.append(success_rate)
        all_inference_freqs.append(task_result['avg_inference_frequency_hz'])
        all_latencies.append(task_result['avg_latency_ms'])
        all_stabilities.append(task_result['avg_stability_score'])
        all_gpu_peaks.append(task_result['gpu_memory_peak'])
        all_cpu_peaks.append(task_result['cpu_memory_peak'])
        
        parts = task_name.split('-')
        if len(parts) >= 2:
            category = parts[0]
            difficulty = parts[-1] if parts[-1] in ['Easy', 'Medium', 'Hard'] else None
            
            if category not in category_results:
                category_results[category] = []
            category_results[category].append(success_rate)
            
            if difficulty:
                difficulty_results[difficulty].append(success_rate)
    
    summary['overall_metrics'] = {
        'model_size': f"{data['model_size']}",
        'average_success_rate': float(np.mean(all_success_rates)),
        'average_inference_frequency_hz': float(np.mean(all_inference_freqs)),
        'average_latency_ms': float(np.mean(all_latencies)),
        'average_stability_score': float(np.mean(all_stabilities)),
        'total_tasks': len(data['results']),
        'successful_tasks': sum(1 for sr in all_success_rates if sr > 50),
        'average_gpu_memory_peak': float(np.mean(all_gpu_peaks)),
        'average_cpu_memory_peak': float(np.mean(all_cpu_peaks)),
    }
    
    for category, rates in category_results.items():
        summary['category_breakdown'][category] = {
            'average_success_rate': float(np.mean(rates)),
            'num_tasks': len(rates),
            'max_success_rate': float(np.max(rates)),
            'min_success_rate': float(np.min(rates))
        }
    
    for difficulty, rates in difficulty_results.items():
        if rates:
            summary['difficulty_breakdown'][difficulty] = {
                'average_success_rate': float(np.mean(rates)),
                'num_tasks': len(rates)
            }
    
    summary_file = output_dir / f'summary_{test_type}_{session_id}.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to: {summary_file}")
    print(f"Overall Success Rate: {summary['overall_metrics']['average_success_rate']:.2f}%")
    print(f"Average Inference Frequency: {summary['overall_metrics']['average_inference_frequency_hz']:.2f} Hz")
    print(f"Average Latency: {summary['overall_metrics']['average_latency_ms']:.2f} ms")
    print(f"Average Stability Score: {summary['overall_metrics']['average_stability_score']:.4f}")
    print(f"Average GPU Memory Peak: {summary['overall_metrics']['average_gpu_memory_peak']:.2f} MB")
    print(f"Average CPU Memory Peak: {summary['overall_metrics']['average_cpu_memory_peak']:.2f} MB")

def save_task_result(task_result, config, param_info, is_first_task=False, test_type='all'):
    """Save task result to JSON file."""
    output_dir = Path(config['experiment']['save_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    session_id = config.get('session_id')
    results_file = output_dir / f'results_{test_type}_{session_id}.json'
    
    if results_file.exists():
        with open(results_file, 'r') as f:
            results_data = json.load(f)
    else:
        serializable_config = make_json_serializable(config)
        results_data = {
            'session_id': session_id,
            'timestamp': datetime.now().isoformat(),
            'config': serializable_config,
            'model_size': f"{param_info['total'] * 4 / 1024**2:.1f} MB (assuming float32)",
            'results': []
        }
    
    serializable_task_result = make_json_serializable(task_result)
    results_data['results'].append(serializable_task_result)
    
    temp_file = output_dir / f'results_{test_type}_{session_id}.json.tmp'
    with open(temp_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    temp_file.replace(results_file)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config_spatialvla.yaml')
    parser.add_argument('--tasks', nargs='+', help='Specific tasks to run')
    parser.add_argument('--test-type', choices=['capability', 'stress', 'all'], default='all')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Generate unique session ID
    config['session_id'] = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    np.random.seed(config['experiment']['random_seed'])
    torch.manual_seed(config['experiment']['random_seed'])
    torch.cuda.manual_seed(config['experiment']['random_seed'])
    
    # Create SpatialVLA policy
    model, processor = create_spatialvla_policy(config)
    policy = SpatialVLAPolicy(model, processor, config)
    param_info = count_parameters(policy)
    
    if args.tasks:
        tasks = args.tasks
    else:
        tasks = []
        if args.test_type in ['capability', 'all']:
            tasks.extend(config['tasks']['capability'])
        if args.test_type in ['stress', 'all']:
            tasks.extend(config['tasks']['stress'])
    
    for idx, task in enumerate(tqdm(tasks, desc="Overall Progress")):
        result = evaluate_task(task, policy, config)
        save_task_result(result, config, param_info, is_first_task=(idx == 0), test_type=args.test_type)

    generate_conclusion(config, args.test_type)

if __name__ == "__main__":
    main()