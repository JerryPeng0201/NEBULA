import gymnasium as gym
import numpy as np
import sapien

from nebula.core.simulation.engine import BaseEnv
import os
import imageio.v2 as imageio

from collections import defaultdict
import tyro
from dataclasses import dataclass
from typing import List, Optional, Annotated, Union
import tqdm
@dataclass
class Args:
    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = "PushCube-v1"
    """The environment ID of the task you want to simulate"""

    obs_mode: Annotated[str, tyro.conf.arg(aliases=["-o"])] = "rgb"
    """Observation mode"""

    robot_uids: Annotated[Optional[str], tyro.conf.arg(aliases=["-r"])] = None
    """Robot UID(s) to use. Can be a comma separated list of UIDs or empty string to have no agents. If not given then defaults to the environments default robot"""

    sim_backend: Annotated[str, tyro.conf.arg(aliases=["-b"])] = "auto"
    """Which simulation backend to use. Can be 'auto', 'cpu', 'gpu'"""

    render_backend: Annotated[str, tyro.conf.arg(aliases=["-rb"])] = "gpu"
    """Which render backend to use. Can be 'gpu', 'cpu', 'none'"""

    reward_mode: Optional[str] = None
    """Reward mode"""

    num_envs: Annotated[int, tyro.conf.arg(aliases=["-n"])] = 1
    """Number of environments to run."""

    control_mode: Annotated[Optional[str], tyro.conf.arg(aliases=["-c"])] = None
    """Control mode"""

    render_mode: str = "rgb_array"
    """Render mode"""

    shader: str = "default"
    """Change shader used for all cameras in the environment for rendering. Default is 'minimal' which is very fast. Can also be 'rt' for ray tracing and generating photo-realistic renders. Can also be 'rt-fast' for a faster but lower quality ray-traced renderer"""

    record_dir: Optional[str] = None
    """Directory to save recordings"""

    pause: Annotated[bool, tyro.conf.arg(aliases=["-p"])] = False
    """If using human render mode, auto pauses the simulation upon loading"""

    quiet: bool = True
    """Disable verbose output."""

    seed: Annotated[Optional[Union[int, List[int]]], tyro.conf.arg(aliases=["-s"])] = None
    """Seed(s) for random actions and simulator. Can be a single integer or a list of integers. Default is None (no seeds)"""
def env_sim(args: Args,env_kwargs,version: str):
    verbose = not args.quiet
    env: BaseEnv = gym.make(
                args.env_id,
                **env_kwargs
            )
    img_data = defaultdict(list)
    record_dir = args.record_dir
    
    if verbose:
        print("Observation space", env.observation_space)
        print("Action space", env.action_space)
        if env.unwrapped.agent is not None:
            print("Control mode", env.unwrapped.control_mode)
        print("Reward mode", env.unwrapped.reward_mode)
    
    obs, _ = env.reset(seed=args.seed, options=dict(reconfigure=True))
    
    for cam_name in obs['sensor_data'].keys():
        img_data[cam_name].append(obs['sensor_data'][cam_name]['rgb'])
    
    
    if args.seed is not None and env.action_space is not None:
            env.action_space.seed(args.seed[0])
    if args.render_mode == "human":
        viewer = env.render()
        if isinstance(viewer, sapien.utils.Viewer):
            viewer.paused = args.pause
        env.render()
    
    while True:
        action = env.action_space.sample() if env.action_space is not None else None
        obs, reward, terminated, truncated, info = env.step(action)
        for cam_name in obs['sensor_data'].keys():
            img_data[cam_name].append(obs['sensor_data'][cam_name]['rgb'])
        if verbose:
            print("reward", reward)
            print("terminated", terminated)
            print("truncated", truncated)
            print("info", info)
        if args.render_mode == "human":
            env.render()
        if args.render_mode is None or args.render_mode != "human":
            if (terminated | truncated).any():
                break
    env.close()

    if record_dir:
        record_dir = os.path.join(args.record_dir, args.env_id, str(env_kwargs["post_process_method"][0].split("+")[0]),version)
        os.makedirs(record_dir, exist_ok=True)
        for cam_name, frames in img_data.items():

            video_path = os.path.join(record_dir, f"{cam_name}.mp4")
            # save video
    
            imgs = [f[0].cpu().numpy().astype(np.uint8) for f in frames]
            imageio.mimsave(video_path, imgs, fps=30, codec="libx264")

def main(args: Args):
    if args.render_mode == "none":
        args.render_mode = None
    np.set_printoptions(suppress=True, precision=3)
    
    if isinstance(args.seed, int):
        args.seed = [args.seed]
    if args.seed is not None:
        np.random.seed(args.seed[0])
    parallel_in_single_scene = args.render_mode == "human"
    if args.render_mode == "human" and args.obs_mode in ["sensor_data", "rgb", "rgbd", "depth", "point_cloud"]:
        print("Disabling parallel single scene/GUI render as observation mode is a visual one. Change observation mode to state or state_dict to see a parallel env render")
        parallel_in_single_scene = False
    if args.render_mode == "human" and args.num_envs == 1:
        parallel_in_single_scene = False
    env_kwargs = dict(
        obs_mode=args.obs_mode,
        reward_mode=args.reward_mode,
        control_mode=args.control_mode,
        render_mode=args.render_mode,
        sensor_configs=dict(shader_pack=args.shader),
        human_render_camera_configs=dict(shader_pack=args.shader),
        viewer_camera_configs=dict(shader_pack=args.shader),
        num_envs=args.num_envs,
        sim_backend=args.sim_backend,
        render_backend=args.render_backend,
        enable_shadow=True,
        parallel_in_single_scene=parallel_in_single_scene,
    )
    if args.robot_uids is not None:
        env_kwargs["robot_uids"] = tuple(args.robot_uids.split(","))
        if len(env_kwargs["robot_uids"]) == 1:
            env_kwargs["robot_uids"] = env_kwargs["robot_uids"][0]
    
    latency_config = {
        "v1": ["latency+5.0"],
        "v2": ["latency+15.0"],
        "v3": ["latency+25.0"]
    }
    packet_drop_config = {
        "v1": ["packet_drop+0.1"],
        "v2": ["packet_drop+0.3"],
        "v3": ["packet_drop+0.5"]
    }

    for config in tqdm.tqdm([latency_config, packet_drop_config]):
        for version in config:
            env_kwargs["post_process_method"] = config[version]
            print(f"Running env with post processing: {env_kwargs['post_process_method']}")
            env_sim(args,env_kwargs,version)



if __name__ == "__main__":
    parsed_args = tyro.cli(Args)
    main(parsed_args)
