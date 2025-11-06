
from typing import Annotated
from torch import _assert
import tyro
from dataclasses import dataclass
import glob
import os
import json
import collections
import shutil
import h5py
import tqdm
@dataclass
class Args:
    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = "ControlEasy-PlaceSphere"
    """the environment id to reformat demos for"""
    record_dir: str = "demos"
    """directory for recorded data and optionally videos"""
    num_episodes: int = 100
    """number of episodes to reformat"""
    target_id: str = "0"
    """the name of the organized folder for all collections."""
    only_success: bool = False
    """whether to only reformat successful episodes."""

def parse_args() -> Args:
    return tyro.cli(Args)




def main(args: Args):
    # add path together
    target_dir = os.path.join(args.record_dir, args.env_id, "teleop")
    print(f"Reformatting {args.num_episodes} episodes of {args.env_id} into {target_dir}/{args.target_id}/")

    # total episode metadata
    total_episode_metadata = {}
    g_id = 0

    # Find all recorded metadata json files
    recorded_json_files = glob.glob(os.path.join(target_dir, "*/metadata.json"))
    recorded_json_files = sorted(recorded_json_files)
    sub_collection_map = collections.defaultdict(list)
    for json_file in recorded_json_files:
        print(f"Found recorded metadata json file: {json_file}")
        sub_idx = int(os.path.basename(os.path.dirname(json_file)))
        with open(json_file, "r") as f:
            metadata = json.load(f)
            # Process metadata as needed
            assert "episodes" in metadata, f"Invalid metadata file: {json_file}"
            assert type(metadata["episodes"]) == list, f"Invalid episodes format in metadata file: {json_file}"
            assert len(metadata["episodes"]) > 0, f"Empty episodes list in metadata file: {json_file}"
            assert len(metadata["episodes"]) <= args.num_episodes, f"More episodes than expected in metadata file: {json_file}"
            for episode in metadata["episodes"]:
                if args.only_success and not episode.get("success", False):
                    continue
                total_episode_metadata[(g_id, sub_idx)] = episode
                sub_collection_map[sub_idx].append((g_id, sub_idx))
                g_id += 1
                if g_id >= args.num_episodes:
                    break

    if g_id < args.num_episodes-1:
        print(f"Warning: only found {g_id} successful episodes, less than requested {args.num_episodes}.")
   
    # Get json template
    template_json_path = recorded_json_files[0]
    with open(template_json_path, "r") as f:
        template_metadata = json.load(f)
        template_metadata["episodes"] = []


    # ensure target directory exists
    target_collection_dir = os.path.join(target_dir, 'tmp_'+args.target_id)
    os.makedirs(target_collection_dir, exist_ok=True)

    # make target video folders
    for mode in ["rgb", "depth", "segmentation"]:
        for view in ['back_left_view', 'back_right_view', 'base_view', 'front_left_view', 'front_right_view','hand_view']:
            os.makedirs(os.path.join(target_collection_dir, view, mode), exist_ok=True)
    
    # find all recorded h5 files
    recorded_h5_files = glob.glob(os.path.join(target_dir, "*/trajectory.h5"))
    h5_path_map = {}
    for h5_file in recorded_h5_files:
        sub_idx = int(os.path.basename(os.path.dirname(h5_file)))
        h5_path_map[sub_idx] = h5_file

    # create target h5 file
    target_h5_path = os.path.join(target_collection_dir, "tmp_trajectory.h5")
    target_h5 = h5py.File(target_h5_path, "w")
    
    # write new episode data and h5 data
    for sub_idx in tqdm.tqdm(sub_collection_map.keys(), desc="Processing sub-collections"):
        episode_list = sub_collection_map[sub_idx]
        # load source h5 file
        source_h5 = h5py.File(h5_path_map[sub_idx], "r")
        for (g_id, sub_idx) in tqdm.tqdm(episode_list, desc=f"Reformatting sub-collection {sub_idx}"):
            episode_info = total_episode_metadata.get((g_id, sub_idx))

            # copy videos
            video_src_path = episode_info["videos"]

            for key, path in video_src_path.items():
                mode_path = str(key.split('_')[-1])
                view_path = '_'.join(key.split('_')[:-2]) + '_view'
                real_dst_path = os.path.join(target_collection_dir, view_path, mode_path, f"{g_id}.mp4")
                
                copy_src_path = os.path.join(target_dir, *path.split('/')[1::])
                shutil.copyfile(copy_src_path, real_dst_path)

                dst_path = os.path.join(args.env_id, args.target_id, view_path, mode_path, f"{g_id}.mp4")
                episode_info["videos"][key] = dst_path
            
            # copy h5 data
            source_h5.copy(f"traj_{episode_info['episode_id']}", target_h5, name=f"traj_{g_id}")

            episode_info["episode_id"] = g_id
            
            # append to new metadata
            template_metadata["episodes"].append(episode_info)
        source_h5.close()
    target_h5.close()

    # write metadata json
    target_metadata_path = os.path.join(target_collection_dir, "tmp_metadata.json")
    with open(target_metadata_path, "w") as f:
        json.dump(template_metadata, f, indent=2)

    return

if __name__ == "__main__":
    main(parse_args())