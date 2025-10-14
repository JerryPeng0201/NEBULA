import sys
sys.path.append('./Isaac-GR00T')
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
import psutil

from gr00t.experiment.data_config import load_data_config
from gr00t.model.policy import Gr00tPolicy

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

def create_gr00t_policy(config):
    data_config = load_data_config(config['model']['data_config_name'])
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()
    
    policy = Gr00tPolicy(
        model_path=config['model']['model_path'],
        modality_config=modality_config,
        modality_transform=modality_transform,
        embodiment_tag=config['model']['embodiment_tag'],
        denoising_steps=config['model']['denoising_steps'],
    )
    return policy

def nebula_to_gr00t_obs(nebula_obs, task_description, info=None):
    base_camera = nebula_obs['sensor_data']['base_camera']['rgb'].detach().cpu().numpy()
    hand_camera = nebula_obs['sensor_data']['hand_camera']['rgb'].detach().cpu().numpy()
    
    if info is not None and 'task_instruction' in info:
        task_description = info['task_instruction']
    
    qpos = nebula_obs['agent']['qpos'].detach().cpu().numpy()
    single_arm = qpos[0, :7]
    gripper = qpos[0, 7:9]
    
    return {
        "video.base_camera": base_camera,
        "video.hand_camera": hand_camera,
        "state.single_arm": single_arm[np.newaxis, ...],
        "state.gripper": gripper[np.newaxis, ...],
        "annotation.human.task_description": [task_description],
    }

def gr00t_to_nebula_action(gr00t_action):
    arm_actions = gr00t_action['action.single_arm']
    gripper_actions = gr00t_action['action.gripper']
    
    nebula_actions = []
    for i in range(arm_actions.shape[0]):
        arm_action = arm_actions[i]
        gripper_action = gripper_actions[i]
        
        if np.isscalar(gripper_action):
            gripper_action = np.array([gripper_action])
        
        combined_action = np.concatenate([arm_action, gripper_action])
        nebula_actions.append(combined_action)
    
    return nebula_actions

def run_episode(env, policy, env_id, config, episode_idx, save_video=False, video_dir=None):
    obs, _ = env.reset(seed=episode_idx + config['experiment']['base_seed'])
    action_history = []
    episode_inference_times = []
    video_frames = []
    global_steps = 0
    task_description_record = ""
    done = False
    
    if hasattr(env, 'evaluate'):
        info = env.evaluate()
    else:
        info = {}
    
    while global_steps < config['experiment']['max_episode_steps'] and not done:
        if hasattr(env.unwrapped, 'get_task_instruction'):
            task_description = env.unwrapped.get_task_instruction()
        else:
            task_description = TASK_DESCRIPTIONS.get(env_id, "Complete the task")

        # record task description only if it changes
        if task_description_record.endswith(task_description) == False:
            if task_description_record != "":
                task_description_record += " THEN "
            task_description_record += task_description
        
        gr00t_obs = nebula_to_gr00t_obs(obs, task_description, info)
        
        start_time = time.perf_counter()
        gr00t_action = policy.get_action(gr00t_obs)
        inference_time = time.perf_counter() - start_time
        episode_inference_times.append(inference_time)
        
        nebula_actions = gr00t_to_nebula_action(gr00t_action)
        
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
        'stability_score': calculate_stability_score(action_history)
    }

def evaluate_task(env_id, policy, config):
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
    """Generate high-level summary of all test results"""
    output_dir = Path(config['experiment']['save_dir'])
    session_id = config.get('session_id')
    results_file = output_dir / f'results_{test_type}_{session_id}.json'
    
    if not results_file.exists():
        print(f"No results file found: {results_file}")
        return
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    # Initialize summary structure
    summary = {
        'session_id': session_id,
        'test_type': test_type,
        'timestamp': datetime.now().isoformat(),
        'model_name': config.get('figure', {}).get('model_name', 'Model'),
        'overall_metrics': {},
        'category_breakdown': {},
        'difficulty_breakdown': {}
    }
    
    # Aggregate results
    all_success_rates = []
    all_inference_freqs = []
    all_latencies = []
    all_stabilities = []
    
    category_results = {}
    difficulty_results = {'Easy': [], 'Medium': [], 'Hard': []}
    
    for task_result in data['results']:
        task_name = task_result['task']
        success_rate = task_result['success_rate']
        
        all_success_rates.append(success_rate)
        all_inference_freqs.append(task_result['avg_inference_frequency_hz'])
        all_latencies.append(task_result['avg_latency_ms'])
        all_stabilities.append(task_result['avg_stability_score'])
        
        # Parse category and difficulty
        parts = task_name.split('-')
        if len(parts) >= 2:
            category = parts[0]
            difficulty = parts[-1] if parts[-1] in ['Easy', 'Medium', 'Hard'] else None
            
            # Category breakdown
            if category not in category_results:
                category_results[category] = []
            category_results[category].append(success_rate)
            
            # Difficulty breakdown
            if difficulty:
                difficulty_results[difficulty].append(success_rate)
    
    # Calculate overall metrics
    summary['overall_metrics'] = {
        'average_success_rate': float(np.mean(all_success_rates)),
        'average_inference_frequency_hz': float(np.mean(all_inference_freqs)),
        'average_latency_ms': float(np.mean(all_latencies)),
        'average_stability_score': float(np.mean(all_stabilities)),
        'total_tasks': len(data['results']),
        'successful_tasks': sum(1 for sr in all_success_rates if sr > 50)
    }
    
    # Category breakdown
    for category, rates in category_results.items():
        summary['category_breakdown'][category] = {
            'average_success_rate': float(np.mean(rates)),
            'num_tasks': len(rates),
            'max_success_rate': float(np.max(rates)),
            'min_success_rate': float(np.min(rates))
        }
    
    # Difficulty breakdown
    for difficulty, rates in difficulty_results.items():
        if rates:
            summary['difficulty_breakdown'][difficulty] = {
                'average_success_rate': float(np.mean(rates)),
                'num_tasks': len(rates)
            }
    
    # Save summary
    summary_file = output_dir / f'summary_{test_type}_{session_id}.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to: {summary_file}")
    print(f"Overall Success Rate: {summary['overall_metrics']['average_success_rate']:.2f}%")
    print(f"Average Inference Frequency: {summary['overall_metrics']['average_inference_frequency_hz']:.2f} Hz")
    print(f"Average Latency: {summary['overall_metrics']['average_latency_ms']:.2f} ms")

def save_task_result(task_result, config, is_first_task=False, test_type='all'):
    output_dir = Path(config['experiment']['save_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    session_id = config.get('session_id')
    results_file = output_dir / f'results_{test_type}_{session_id}.json'
    
    # Load or create results structure
    if results_file.exists():
        with open(results_file, 'r') as f:
            results_data = json.load(f)
    else:
        serializable_config = make_json_serializable(config)
        results_data = {
            'session_id': session_id,
            'timestamp': datetime.now().isoformat(),
            'config': serializable_config,
            'results': []
        }
    
    # Append task result
    serializable_task_result = make_json_serializable(task_result)
    results_data['results'].append(serializable_task_result)
    
    # Write atomically
    temp_file = output_dir / f'results_{test_type}_{session_id}.json.tmp'
    with open(temp_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    temp_file.replace(results_file)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
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
    
    policy = create_gr00t_policy(config)
    
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
        save_task_result(result, config, is_first_task=(idx == 0),test_type=args.test_type)

    generate_conclusion(config, args.test_type)

if __name__ == "__main__":
    main()