# modified from Maniskill/examples/baselines/diffusion_policy/train.py

ALGO_NAME = 'BC_Diffusion_state_UNet'

import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from diffusion_policy.evaluate import evaluate
from collections import defaultdict

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'diffusion_policy'))
import yaml

from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import RandomSampler, BatchSampler
from torch.utils.data.dataloader import DataLoader
from diffusion_policy.utils import IterationBasedBatchSampler, worker_init_fn
from diffusion_policy.make_env import make_eval_envs
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from diffusion_policy.conditional_unet1d import ConditionalUnet1D
from dataclasses import dataclass, field
from typing import Optional, List
import tyro
import json
import glob

@dataclass
class Args:
    exp_name: Optional[str] = None
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "Nebula"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Dataset loading options - use either demo_path OR (dataset_root + task_names)
    demo_path: Optional[str] = None
    """the path to a single dataset folder (legacy mode)"""


    with open('../../config.yaml', 'r') as file:
            nebula_config = yaml.safe_load(file)
            dataset_dir = nebula_config['experiment']['dataset_root']
    
    dataset_root: str = os.path.expanduser("~/mnt_hpc_data/alpha")
    """root directory containing all task datasets"""
    task_names: List[str] = field(default_factory=lambda: [
        "Control-PlaceSphere-Easy", "Control-PushCube-Easy", "Control-StackCube-Easy", 
        "Control-PegInsertionSide-Medium", "Control-PlaceSphere-Medium", "Control-StackCube-Medium", 
        "Control-PlaceSphere-Hard", "Control-StackCube-Hard", 
        "Perception-PlaceBiggerSphere-Easy", "Perception-PlaceRedSphere-Easy", "Perception-PlaceSphere-Easy", 
        "Perception-PlaceDiffCubes-Medium", "Perception-PlaceRedT-Medium", "Perception-PlaceWhitePeg-Medium", 
        "Perception-PlacePeg-Hard", "Perception-PlaceRedT-Hard", "Perception-PlaceRightCubes-Hard", 
        "DynamicEasy-PressSwitch", "DynamicMedium-PickSlidingCube", "DynamicHard-ColorSwitchPickCube", "DynamicHard-ShapeSwitchPickCube",
        "SpatialReferenceEasy-MoveCube", "SpatialReferenceEasy-PickCube"
    ])
    """list of task names to load data from"""
    
    # Evaluation environment (can be different from training tasks)
    eval_env_id: Optional[str] = None
    """environment ID for evaluation. If None, uses the first task from task_names"""
    
    num_demos: Optional[int] = None
    """number of trajectories to load from the demo dataset"""
    total_iters: int = 1_000_000
    """total timesteps of the experiment"""
    batch_size: int = 1024
    """the batch size of sample from the replay memory"""

    # Diffusion Policy specific arguments
    lr: float = 1e-4
    """the learning rate of the diffusion policy"""
    obs_horizon: int = 2 # Seems not very important in ManiSkill, 1, 2, 4 work well
    act_horizon: int = 8 # Seems not very important in ManiSkill, 4, 8, 15 work well
    pred_horizon: int = 16 # 16->8 leads to worse performance, maybe it is like generate a half image; 16->32, improvement is very marginal
    diffusion_step_embed_dim: int = 64 # not very important
    unet_dims: List[int] = field(default_factory=lambda: [64, 128, 256]) # default setting is about ~4.5M params
    n_groups: int = 8 # jigu says it is better to let each group have at least 8 channels; it seems 4 and 8 are similar

    # Environment/experiment specific arguments
    max_episode_steps: Optional[int] = None
    """Change the environments' max_episode_steps to this value. Sometimes necessary if the demonstrations being imitated are too short. Typically the default
    max episode steps of environments in ManiSkill are tuned lower so reinforcement learning agents can learn faster."""
    log_freq: int = 1000
    """the frequency of logging the training metrics"""
    eval_freq: int = 5000
    """the frequency of evaluating the agent on the evaluation environments"""
    save_freq: Optional[int] = None
    """the frequency of saving the model checkpoints. By default this is None and will only save checkpoints based on the best evaluation metrics."""
    num_eval_episodes: int = 100
    """the number of episodes to evaluate the agent on"""
    num_eval_envs: int = 10
    """the number of parallel environments to evaluate the agent on"""
    sim_backend: str = "physx_cpu"
    """the simulation backend to use for evaluation environments. can be "cpu" or "gpu"""
    num_dataload_workers: int = 0
    """the number of workers to use for loading the training data in the torch dataloader"""
    control_mode: str = 'pd_joint_delta_pos'
    """the control mode to use for the evaluation environments. Must match the control mode of the demonstration dataset."""

    # additional tags/configs for logging purposes to wandb and shared comparisons with other algorithms
    demo_type: Optional[str] = None


class SmallDemoDataset_DiffusionPolicy(Dataset): # Load everything into GPU memory
    def __init__(self, data_config, device, num_traj):
        # Import the modified load_demo_dataset function
        from diffusion_policy.utils import load_demo_dataset
        
        # Load trajectories using the new function that handles your dataset structure
        trajectories = load_demo_dataset(data_config, num_traj=num_traj, concat=False)
        # trajectories['observations'] is a list of np.ndarray (L+1, obs_dim)
        # trajectories['actions'] is a list of np.ndarray (L, act_dim)

        # Convert to GPU tensors
        for k, v in trajectories.items():
            for i in range(len(v)):
                trajectories[k][i] = torch.Tensor(v[i]).to(device)

        # Pre-compute all possible (traj_idx, start, end) tuples, this is very specific to Diffusion Policy
        if 'delta_pos' in args.control_mode or args.control_mode == 'base_pd_joint_vel_arm_pd_joint_vel':
            self.pad_action_arm = torch.zeros((trajectories['actions'][0].shape[1]-1,), device=device)
            # to make the arm stay still, we pad the action with 0 in 'delta_pos' control mode
            # gripper action needs to be copied from the last action
        
        self.obs_horizon, self.pred_horizon = obs_horizon, pred_horizon = args.obs_horizon, args.pred_horizon
        self.slices = []
        num_traj = len(trajectories['actions'])
        total_transitions = 0
        
        for traj_idx in range(num_traj):
            L = trajectories['actions'][traj_idx].shape[0]
            assert trajectories['observations'][traj_idx].shape[0] == L + 1
            total_transitions += L

            # |o|o|                             observations: 2
            # | |a|a|a|a|a|a|a|a|               actions executed: 8
            # |p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16
            pad_before = obs_horizon - 1
            # Pad before the trajectory, so the first action of an episode is in "actions executed"
            # obs_horizon - 1 is the number of "not used actions"
            pad_after = pred_horizon - obs_horizon
            # Pad after the trajectory, so all the observations are utilized in training
            # Note that in the original code, pad_after = act_horizon - 1, but I think this is not the best choice
            self.slices += [
                (traj_idx, start, start + pred_horizon) for start in range(-pad_before, L - pred_horizon + pad_after)
            ]  # slice indices follow convention [start, end)

        print(f"Total transitions: {total_transitions}, Total obs sequences: {len(self.slices)}")
        self.trajectories = trajectories

    def __getitem__(self, index):
        traj_idx, start, end = self.slices[index]
        L, act_dim = self.trajectories['actions'][traj_idx].shape

        obs_seq = self.trajectories['observations'][traj_idx][max(0, start):start+self.obs_horizon]
        # start+self.obs_horizon is at least 1
        act_seq = self.trajectories['actions'][traj_idx][max(0, start):end]
        if start < 0: # pad before the trajectory
            obs_seq = torch.cat([obs_seq[0].repeat(-start, 1), obs_seq], dim=0)
            act_seq = torch.cat([act_seq[0].repeat(-start, 1), act_seq], dim=0)
        if end > L: # pad after the trajectory
            gripper_action = act_seq[-1, -1]
            pad_action = torch.cat((self.pad_action_arm, gripper_action[None]), dim=0)
            act_seq = torch.cat([act_seq, pad_action.repeat(end-L, 1)], dim=0)
            # making the robot (arm and gripper) stay still
        assert obs_seq.shape[0] == self.obs_horizon and act_seq.shape[0] == self.pred_horizon
        return {
            'observations': obs_seq,
            'actions': act_seq,
        }

    def __len__(self):
        return len(self.slices)


class Agent(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        self.obs_horizon = args.obs_horizon
        self.act_horizon = args.act_horizon
        self.pred_horizon = args.pred_horizon
        assert len(env.single_observation_space.shape) == 2 # (obs_horizon, obs_dim)
        assert len(env.single_action_space.shape) == 1 # (act_dim, )
        assert (env.single_action_space.high == 1).all() and (env.single_action_space.low == -1).all()
        # denoising results will be clipped to [-1,1], so the action should be in [-1,1] as well
        self.act_dim = env.single_action_space.shape[0]

        self.noise_pred_net = ConditionalUnet1D(
            input_dim=self.act_dim, # act_horizon is not used (U-Net doesn't care)
            global_cond_dim=np.prod(env.single_observation_space.shape), # obs_horizon * obs_dim
            diffusion_step_embed_dim=args.diffusion_step_embed_dim,
            down_dims=args.unet_dims,
            n_groups=args.n_groups,
        )
        self.num_diffusion_iters = 100
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            beta_schedule='squaredcos_cap_v2', # has big impact on performance, try not to change
            clip_sample=True, # clip output to [-1,1] to improve stability
            prediction_type='epsilon' # predict noise (instead of denoised action)
        )

    def compute_loss(self, obs_seq, action_seq):
        B = obs_seq.shape[0]

        # observation as FiLM conditioning
        obs_cond = obs_seq.flatten(start_dim=1) # (B, obs_horizon * obs_dim)

        # sample noise to add to actions
        noise = torch.randn((B, self.pred_horizon, self.act_dim), device=device)

        # sample a diffusion iteration for each data point
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (B,), device=device
        ).long()

        # add noise to the clean images(actions) according to the noise magnitude at each diffusion iteration
        # (this is the forward diffusion process)
        noisy_action_seq = self.noise_scheduler.add_noise(
            action_seq, noise, timesteps)

        # predict the noise residual
        noise_pred = self.noise_pred_net(
            noisy_action_seq, timesteps, global_cond=obs_cond)

        return F.mse_loss(noise_pred, noise)

    def get_action(self, obs_seq):
        # init scheduler
        # self.noise_scheduler.set_timesteps(self.num_diffusion_iters)
        # set_timesteps will change noise_scheduler.timesteps is only used in noise_scheduler.step()
        # noise_scheduler.step() is only called during inference
        # if we use DDPM, and inference_diffusion_steps == train_diffusion_steps, then we can skip this

        # obs_seq: (B, obs_horizon, obs_dim)
        B = obs_seq.shape[0]
        with torch.no_grad():
            obs_cond = obs_seq.flatten(start_dim=1) # (B, obs_horizon * obs_dim)

            # initialize action from Guassian noise
            noisy_action_seq = torch.randn((B, self.pred_horizon, self.act_dim), device=obs_seq.device)

            for k in self.noise_scheduler.timesteps:
                # predict noise
                noise_pred = self.noise_pred_net(
                    sample=noisy_action_seq,
                    timestep=k,
                    global_cond=obs_cond,
                )

                # inverse diffusion step (remove noise)
                noisy_action_seq = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=noisy_action_seq,
                ).prev_sample

        # only take act_horizon number of actions
        start = self.obs_horizon - 1
        end = start + self.act_horizon
        return noisy_action_seq[:, start:end] # (B, act_horizon, act_dim)


def save_ckpt(run_name, tag):
    os.makedirs(f'/HDD1/embodied_ai/checkpoints/Nebula_finetune/dp/runs/{run_name}/checkpoints', exist_ok=True)
    ema.copy_to(ema_agent.parameters())
    torch.save({
        'agent': agent.state_dict(),
        'ema_agent': ema_agent.state_dict(),
    }, f'/HDD1/embodied_ai/checkpoints/Nebula_finetune/dp/runs/{run_name}/checkpoints/{tag}.pt')


def prepare_dataset_config(args):
    """
    Prepare dataset configuration based on provided arguments
    """
    # Check if using legacy single path mode
    if args.demo_path is not None:
        print(f"Using legacy mode - loading data from single path: {args.demo_path}")
        return args.demo_path
    else:
        # Multi-task mode (default)
        print(f"Loading data from {len(args.task_names)} tasks in {args.dataset_root}")
        print(f"Tasks: {args.task_names}")
        return {
            'dataset_root': args.dataset_root,
            'task_names': args.task_names
        }


def get_evaluation_env_id(args):
    """
    Get the environment ID to use for evaluation
    """
    if args.eval_env_id is not None:
        return args.eval_env_id
    elif args.demo_path is not None:
        # Legacy mode - try to infer from demo_path or use default
        return "PegInsertionSide-v0"  # fallback
    else:
        # Use first task name as env_id
        return args.task_names[0]


def validate_dataset_config(data_config):
    """Validate that the dataset configuration is valid"""
    if isinstance(data_config, str):
        # Single path mode
        if not os.path.exists(data_config):
            raise ValueError(f"Dataset path does not exist: {data_config}")
        
        # Check for H5 files
        h5_files = glob.glob(os.path.join(data_config, "*.h5"))
        if not h5_files:
            raise ValueError(f"No H5 files found in dataset path: {data_config}")
        
        print(f"Found {len(h5_files)} H5 files in {data_config}")
        
    elif isinstance(data_config, dict):
        # Multi-task mode
        dataset_root = data_config['dataset_root']
        task_names = data_config['task_names']
        
        if not os.path.exists(dataset_root):
            raise ValueError(f"Dataset root does not exist: {dataset_root}")
        
        # Validate each task exists
        for task_name in task_names:
            task_path = os.path.join(dataset_root, task_name)
            if not os.path.exists(task_path):
                raise ValueError(f"Task directory does not exist: {task_path}")
        
        print(f"Validated {len(task_names)} tasks in {dataset_root}")
    
    return True


def extract_control_mode_from_config(data_config):
    """Extract control mode from JSON metadata files"""
    json_files = []
    
    if isinstance(data_config, str):
        # Single path mode
        json_files = glob.glob(os.path.join(data_config, "*.json"))
    elif isinstance(data_config, dict):
        # Multi-task mode - check first subtask of first task
        dataset_root = data_config['dataset_root']
        task_names = data_config['task_names']
        
        if task_names:
            # Look for JSON files in the first task's first subtask
            first_task_path = os.path.join(dataset_root, task_names[0], "motionplanning")
            if os.path.exists(first_task_path):
                subtasks = [d for d in os.listdir(first_task_path) if d.startswith('subtask_')]
                if subtasks:
                    first_subtask = os.path.join(first_task_path, sorted(subtasks)[0])
                    json_files = glob.glob(os.path.join(first_subtask, "*.json"))
    
    if not json_files:
        print("Warning: No JSON files found, cannot verify control mode")
        return None
    
    # Load the first JSON file to get control mode
    json_file = json_files[0]
    try:
        with open(json_file, 'r') as f:
            demo_info = json.load(f)
            
        # Check different possible locations for control mode
        control_mode = None
        if 'env_info' in demo_info and 'env_kwargs' in demo_info['env_info']:
            control_mode = demo_info['env_info']['env_kwargs'].get('control_mode')
        elif 'episodes' in demo_info and len(demo_info['episodes']) > 0:
            control_mode = demo_info['episodes'][0].get('control_mode')
        
        if control_mode:
            print(f"Detected control mode from JSON: {control_mode}")
            return control_mode
        else:
            print("Warning: Control mode not found in JSON metadata")
            return None
            
    except Exception as e:
        print(f"Warning: Failed to read JSON file {json_file}: {e}")
        return None


if __name__ == "__main__":
    args = tyro.cli(Args)
    
    # Get evaluation environment ID from task names or arguments
    eval_env_id = get_evaluation_env_id(args)
    
    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{eval_env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name

    # Prepare and validate dataset configuration
    data_config = prepare_dataset_config(args)
    validate_dataset_config(data_config)
    
    # Try to extract and validate control mode from JSON files
    detected_control_mode = extract_control_mode_from_config(data_config)
    if detected_control_mode and detected_control_mode != args.control_mode:
        print(f"Warning: Control mode mismatch!")
        print(f"  Dataset control mode: {detected_control_mode}")
        print(f"  Args control mode: {args.control_mode}")
        print(f"  Using args control mode: {args.control_mode}")

    # Validate horizon constraints
    assert args.obs_horizon + args.act_horizon - 1 <= args.pred_horizon
    assert args.obs_horizon >= 1 and args.act_horizon >= 1 and args.pred_horizon >= 1

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    env_kwargs = dict(control_mode=args.control_mode, reward_mode="sparse", obs_mode="state", render_mode="rgb_array", human_render_camera_configs=dict(shader_pack="default"))
    assert args.max_episode_steps != None, "max_episode_steps must be specified as imitation learning algorithms task solve speed is dependent on the data you train on"
    env_kwargs["max_episode_steps"] = args.max_episode_steps
    other_kwargs = dict(obs_horizon=args.obs_horizon)
    envs = make_eval_envs(eval_env_id, args.num_eval_envs, args.sim_backend, env_kwargs, other_kwargs, video_dir=f'runs/{run_name}/videos' if args.capture_video else None)

    if args.track:
        import wandb
        config = vars(args)
        config["eval_env_cfg"] = dict(**env_kwargs, num_envs=args.num_eval_envs, env_id=eval_env_id, env_horizon=args.max_episode_steps)
        
        # Add dataset info to wandb config
        if isinstance(data_config, dict):
            config["dataset_info"] = {
                "mode": "multi_task",
                "dataset_root": data_config['dataset_root'],
                "task_names": data_config['task_names'],
                "num_tasks": len(data_config['task_names'])
            }
        else:
            config["dataset_info"] = {
                "mode": "single_path",
                "demo_path": data_config
            }
        
        # Add evaluation environment info
        config["eval_env_id"] = eval_env_id
        
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=config,
            name=run_name,
            save_code=True,
            group="DiffusionPolicy",
            tags=["diffusion_policy", "multi_task" if isinstance(data_config, dict) else "single_task"]
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # dataloader setup
    dataset = SmallDemoDataset_DiffusionPolicy(data_config, device, num_traj=args.num_demos)
    sampler = RandomSampler(dataset, replacement=False)
    batch_sampler = BatchSampler(sampler, batch_size=args.batch_size, drop_last=True)
    batch_sampler = IterationBasedBatchSampler(batch_sampler, args.total_iters)
    train_dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=args.num_dataload_workers,
        worker_init_fn=lambda worker_id: worker_init_fn(worker_id, base_seed=args.seed),
    )
    if args.num_demos is None:
        args.num_demos = len(dataset)

    # agent setup
    agent = Agent(envs, args).to(device)
    optimizer = optim.AdamW(params=agent.parameters(),
        lr=args.lr, betas=(0.95, 0.999), weight_decay=1e-6)

    # Cosine LR schedule with linear warmup
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=args.total_iters,
    )

    # Exponential Moving Average
    # accelerates training and improves stability
    # holds a copy of the model weights
    ema = EMAModel(parameters=agent.parameters(), power=0.75)
    ema_agent = Agent(envs, args).to(device)

    best_eval_metrics = defaultdict(float)
    timings = defaultdict(float)

    # define evaluation and logging functions
    def evaluate_and_save_best(iteration):
        if iteration % args.eval_freq == 0:
            last_tick = time.time()
            ema.copy_to(ema_agent.parameters())
            eval_metrics = evaluate(
                args.num_eval_episodes, ema_agent, envs, device, args.sim_backend
            )
            timings["eval"] += time.time() - last_tick

            print(f"Evaluated {len(eval_metrics['success_at_end'])} episodes")
            for k in eval_metrics.keys():
                eval_metrics[k] = np.mean(eval_metrics[k])
                writer.add_scalar(f"eval/{k}", eval_metrics[k], iteration)
                print(f"{k}: {eval_metrics[k]:.4f}")

            save_on_best_metrics = ["success_once", "success_at_end"]
            for k in save_on_best_metrics:
                if k in eval_metrics and eval_metrics[k] > best_eval_metrics[k]:
                    best_eval_metrics[k] = eval_metrics[k]
                    save_ckpt(run_name, f"best_eval_{k}")
                    print(
                        f"New best {k}_rate: {eval_metrics[k]:.4f}. Saving checkpoint."
                    )
    
    def log_metrics(iteration):
        if iteration % args.log_freq == 0:
            writer.add_scalar(
                "charts/learning_rate", optimizer.param_groups[0]["lr"], iteration
            )
            writer.add_scalar("losses/total_loss", total_loss.item(), iteration)
            for k, v in timings.items():
                writer.add_scalar(f"time/{k}", v, iteration)

    # ---------------------------------------------------------------------------- #
    # Training begins.
    # ---------------------------------------------------------------------------- #
    agent.train()
    pbar = tqdm(total=args.total_iters)
    last_tick = time.time()
    for iteration, data_batch in enumerate(train_dataloader):
        timings["data_loading"] += time.time() - last_tick

        # forward and compute loss
        last_tick = time.time()
        total_loss = agent.compute_loss(
            obs_seq=data_batch["observations"],  # obs_batch_dict['state'] is (B, L, obs_dim)
            action_seq=data_batch["actions"],  # (B, L, act_dim)
        )
        timings["forward"] += time.time() - last_tick

        # backward
        last_tick = time.time()
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        lr_scheduler.step()  # step lr scheduler every batch, this is different from standard pytorch behavior
        timings["backward"] += time.time() - last_tick

        # ema step
        last_tick = time.time()
        ema.step(agent.parameters())
        timings["ema"] += time.time() - last_tick

        # Evaluation
        evaluate_and_save_best(iteration)
        log_metrics(iteration)

        # Checkpoint
        if args.save_freq is not None and iteration % args.save_freq == 0:
            save_ckpt(run_name, str(iteration))
        pbar.update(1)
        pbar.set_postfix({"loss": total_loss.item()})
        last_tick = time.time()

    evaluate_and_save_best(args.total_iters)
    log_metrics(args.total_iters)

    envs.close()
    writer.close()