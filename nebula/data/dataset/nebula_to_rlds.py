"""Utilities to convert NEBULA-Beta trajectories into RLDS SequenceExamples.

The conversion pipeline loads demonstrations stored in the NEBULA Unified Data
Format (HDF5 + JSON metadata) and exports them as TFRecord shards containing
`tf.train.SequenceExample` protos that follow the RLDS step/episode schema.

Example usage from the command line::

    python -m nebula.data.dataset.nebula_to_rlds \
        --data-root ~/datasets/nebula_beta \
        --output-dir ~/datasets/nebula_beta_rlds \
        --tasks Control-PlaceSphere-Easy,Control-PushCube-Easy \
        --max-episodes 100

The resulting directory will contain sharded TFRecord files and a small JSON
manifest describing the conversion. Each SequenceExample stores per-episode
context (task name, language instruction, etc.) and per-step sequence features
matching the RLDS conventions (`observation`, `action`, `reward`, `discount`,
`is_first`, `is_last`, `is_terminal`).

Limitations
----------
* Only RGB/depth images for the `base_camera` and `hand_camera` sensors are
  exported. Additional modalities can be added by extending `_encode_observation`.
* Rewards are optional in the source data. Missing rewards default to zero, and
  discounts default to `1.0` except for the last step (`0.0`).
* The script depends on TensorFlow for serialization and OpenCV (cv2) for image
  encoding; ensure they are installed in the active environment.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import cv2
import h5py
import numpy as np
import tensorflow as tf

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


# ---------------------------------------------------------------------------
# Dataclasses and configuration helpers
# ---------------------------------------------------------------------------


@dataclass
class ConverterConfig:
    """Configuration options for Nebula -> RLDS conversion."""

    data_root: Path
    output_dir: Path
    tasks: Optional[Sequence[str]] = None
    max_episodes: Optional[int] = None
    episodes_per_shard: int = 100
    compression: str = "GZIP"

    @staticmethod
    def from_args(args: argparse.Namespace) -> "ConverterConfig":
        tasks: Optional[Sequence[str]] = None
        if args.tasks:
            tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]

        return ConverterConfig(
            data_root=Path(args.data_root).expanduser().resolve(),
            output_dir=Path(args.output_dir).expanduser().resolve(),
            tasks=tasks,
            max_episodes=args.max_episodes,
            episodes_per_shard=args.episodes_per_shard,
            compression=args.compression.upper() if args.compression else "GZIP",
        )


# ---------------------------------------------------------------------------
# Low-level helpers for TF features and image encoding
# ---------------------------------------------------------------------------


def _bytes_feature(value: bytes) -> tf.train.Feature:
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value: Iterable[float]) -> tf.train.Feature:
    return tf.train.Feature(float_list=tf.train.FloatList(value=list(value)))


def _int_feature(value: Iterable[int]) -> tf.train.Feature:
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list(value)))


def _append_feature(feature_lists: tf.train.FeatureLists, name: str, feature: tf.train.Feature) -> None:
    feature_lists.feature_list[name].feature.add().CopyFrom(feature)


def _encode_image(image: np.ndarray, ext: str = "jpg") -> bytes:
    """Encode an RGB/Depth image array into bytes using OpenCV."""

    if image.dtype != np.uint8:
        # Normalize floats into [0, 255] for visualization-friendly storage
        image = np.clip(image, 0.0, 1.0) if image.dtype == np.float32 else image
        image = (image * 255.0).astype(np.uint8)

    if image.ndim == 2:  # Grayscale/depth
        encode_image = image
    elif image.ndim == 3 and image.shape[2] == 3:
        encode_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        raise ValueError(f"Unsupported image shape for encoding: {image.shape}")

    success, buffer = cv2.imencode(f".{ext}", encode_image)
    if not success:
        raise RuntimeError("Failed to encode image using OpenCV")
    return buffer.tobytes()


# ---------------------------------------------------------------------------
# NEBULA trajectory loading
# ---------------------------------------------------------------------------


def _discover_episodes(data_root: Path, tasks: Optional[Sequence[str]]) -> List[Dict]:
    """Return metadata for all trajectories that match the filter."""

    episode_records: List[Dict] = []

    for task_dir in sorted(data_root.iterdir()):
        if not task_dir.is_dir():
            continue

        task_name = task_dir.name
        if tasks and task_name not in tasks:
            continue

        for phase_name in ("motionplanning", "teleop"):
            phase_dir = task_dir / phase_name
            if not phase_dir.exists():
                continue

            for subtask_dir in sorted(phase_dir.iterdir()):
                if not subtask_dir.is_dir():
                    continue

                h5_files = list(subtask_dir.glob("*.h5"))
                json_files = list(subtask_dir.glob("*.json"))
                if len(h5_files) != 1 or len(json_files) != 1:
                    print(
                        f"[WARN] Skipping {subtask_dir} (expected single h5/json, found {len(h5_files)}/{len(json_files)})"
                    )
                    continue

                h5_path = h5_files[0]
                json_path = json_files[0]

                with open(json_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)

                for episode_meta in metadata.get("episodes", []):
                    episode_records.append(
                        {
                            "task_name": task_name,
                            "phase": phase_name,
                            "subtask": subtask_dir.name,
                            "episode_id": episode_meta["episode_id"],
                            "h5_path": h5_path,
                            "json_path": json_path,
                            "episode_meta": episode_meta,
                            "env_info": metadata.get("env_info", {}),
                        }
                    )

    return episode_records


def _load_episode(record: Dict) -> Dict:
    """Load per-step arrays for a single Nebula trajectory."""

    with h5py.File(record["h5_path"], "r") as hf:
        traj_key = f"traj_{record['episode_id']}"
        if traj_key not in hf:
            raise KeyError(f"Trajectory {traj_key} not found in {record['h5_path']}")

        traj_group = hf[traj_key]
        obs_group = traj_group["obs"]

        # Extract base arrays
        actions = np.array(traj_group["actions"], dtype=np.float32)
        rewards = np.array(traj_group["rewards"], dtype=np.float32) if "rewards" in traj_group else None
        success = np.array(traj_group["success"], dtype=bool) if "success" in traj_group else None
        terminated = np.array(traj_group["terminated"], dtype=bool) if "terminated" in traj_group else None

        sensor_data = obs_group.get("sensor_data", {})
        base_camera = sensor_data.get("base_camera")
        hand_camera = sensor_data.get("hand_camera")

        language_instruction = record["episode_meta"].get("task_instruction", "")

        return {
            "actions": actions,
            "rewards": rewards,
            "success": success,
            "terminated": terminated,
            "language": language_instruction,
            "episode_meta": record["episode_meta"],
            "env_info": record["env_info"],
            "base_camera_rgb": np.array(base_camera["rgb"]) if base_camera and "rgb" in base_camera else None,
            "hand_camera_rgb": np.array(hand_camera["rgb"]) if hand_camera and "rgb" in hand_camera else None,
            "base_camera_depth": np.array(base_camera["depth"]) if base_camera and "depth" in base_camera else None,
            "agent_state": np.array(obs_group["agent"]["qpos"], dtype=np.float32),
            "tcp_pose": np.array(obs_group["extra"]["tcp_pose"], dtype=np.float32) if "extra" in obs_group else None,
        }


# ---------------------------------------------------------------------------
# RLDS SequenceExample construction
# ---------------------------------------------------------------------------


def _episode_to_sequence_example(
    record: Dict,
    episode_data: Dict,
) -> tf.train.SequenceExample:
    """Convert raw episode arrays into a TF SequenceExample."""

    actions = episode_data["actions"]
    num_steps = actions.shape[0]

    rewards = episode_data.get("rewards")
    success = episode_data.get("success")
    terminated = episode_data.get("terminated")
    agent_state = episode_data.get("agent_state")
    tcp_pose = episode_data.get("tcp_pose")
    base_rgb = episode_data.get("base_camera_rgb")
    hand_rgb = episode_data.get("hand_camera_rgb")
    depth_primary = episode_data.get("base_camera_depth")

    if agent_state is None:
        raise ValueError("Agent qpos state is required but missing from episode data")

    language = episode_data.get("language", "").encode("utf-8")
    task_name = record["task_name"].encode("utf-8")
    phase = record.get("phase", "motionplanning")
    episode_id = f"{record['task_name']}_{phase}_{record['subtask']}_{record['episode_id']}".encode("utf-8")

    context_features = {
        "episode_id": _bytes_feature(episode_id),
        "task_name": _bytes_feature(task_name),
        "language_instruction": _bytes_feature(language),
        "num_steps": _int_feature([num_steps]),
        "episode_success": _int_feature([int(bool(record["episode_meta"].get("success", False)))]),
    }

    example = tf.train.SequenceExample(
        context=tf.train.Features(feature=context_features)
    )

    feature_lists = example.feature_lists

    for step_idx in range(num_steps):
        is_first = int(step_idx == 0)
        is_last = int(step_idx == num_steps - 1)
        is_terminal = int(bool(terminated[step_idx])) if terminated is not None else is_last

        reward = rewards[step_idx] if rewards is not None else 0.0
        discount = 0.0 if is_last else 1.0

        _append_feature(feature_lists, "action", _float_feature(actions[step_idx].astype(np.float32)))
        _append_feature(feature_lists, "reward", _float_feature([float(reward)]))
        _append_feature(feature_lists, "discount", _float_feature([float(discount)]))
        _append_feature(feature_lists, "is_first", _int_feature([is_first]))
        _append_feature(feature_lists, "is_last", _int_feature([is_last]))
        _append_feature(feature_lists, "is_terminal", _int_feature([is_terminal]))

        if success is not None:
            _append_feature(feature_lists, "step_success", _int_feature([int(bool(success[step_idx]))]))

        _append_feature(
            feature_lists,
            "observation/state",
            _float_feature(agent_state[min(step_idx, agent_state.shape[0] - 1)]),
        )

        if tcp_pose is not None:
            _append_feature(
                feature_lists,
                "observation/tcp_pose",
                _float_feature(tcp_pose[min(step_idx, tcp_pose.shape[0] - 1)]),
            )

        if base_rgb is not None:
            encoded = _encode_image(base_rgb[min(step_idx, base_rgb.shape[0] - 1)])
            _append_feature(feature_lists, "observation/image_primary", _bytes_feature(encoded))

        if hand_rgb is not None:
            encoded = _encode_image(hand_rgb[min(step_idx, hand_rgb.shape[0] - 1)])
            _append_feature(feature_lists, "observation/image_wrist", _bytes_feature(encoded))

        if depth_primary is not None:
            depth_slice = depth_primary[min(step_idx, depth_primary.shape[0] - 1)].squeeze()
            _append_feature(
                feature_lists,
                "observation/depth_primary",
                _bytes_feature(_encode_image(depth_slice, ext="png")),
            )

        _append_feature(feature_lists, "language_instruction_step", _bytes_feature(language))

    return example


# ---------------------------------------------------------------------------
# Conversion driver
# ---------------------------------------------------------------------------


def convert_dataset(config: ConverterConfig) -> Dict[str, int]:
    """Run the conversion for the configured dataset."""

    episodes = _discover_episodes(config.data_root, config.tasks)
    if config.max_episodes is not None:
        episodes = episodes[: config.max_episodes]

    if not episodes:
        raise ValueError("No episodes found matching the provided filters")

    config.output_dir.mkdir(parents=True, exist_ok=True)

    shard_counters: Dict[str, int] = {"episodes": 0, "steps": 0, "shards": 0}

    options = tf.io.TFRecordOptions(compression_type=config.compression)

    if config.episodes_per_shard <= 0:
        raise ValueError("episodes_per_shard must be a positive integer")

    writer: Optional[tf.io.TFRecordWriter] = None
    shard_index = 0

    iterator = enumerate(episodes)
    if tqdm is not None:
        iterator = tqdm(iterator, total=len(episodes), desc="Converting episodes", unit="ep")

    try:
        for episode_index, record in iterator:
            if episode_index % config.episodes_per_shard == 0:
                if writer is not None:
                    writer.close()
                shard_path = config.output_dir / f"nebula_rlds-{shard_index:05d}.tfrecord"
                writer = tf.io.TFRecordWriter(str(shard_path), options=options)
                shard_index += 1
                shard_counters["shards"] += 1

            episode_data = _load_episode(record)
            example = _episode_to_sequence_example(record, episode_data)
            if writer is None:
                raise RuntimeError("TFRecordWriter was not initialised")
            writer.write(example.SerializeToString())

            shard_counters["episodes"] += 1
            shard_counters["steps"] += episode_data["actions"].shape[0]

    finally:
        if writer is not None:
            writer.close()

    manifest = {
        "config": {
            "data_root": str(config.data_root),
            "tasks": config.tasks,
            "max_episodes": config.max_episodes,
            "episodes_per_shard": config.episodes_per_shard,
            "compression": config.compression,
        },
        "counters": shard_counters,
    }

    manifest_path = config.output_dir / "conversion_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return shard_counters


# ---------------------------------------------------------------------------
# Command-line entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert NEBULA-Beta data to RLDS SequenceExamples")
    parser.add_argument("--data-root", required=True, help="Root directory containing NEBULA tasks")
    parser.add_argument("--output-dir", required=True, help="Directory to write TFRecord shards")
    parser.add_argument("--tasks", default=None, help="Comma-separated task names to include (default: all)")
    parser.add_argument("--max-episodes", type=int, default=None, help="Optional maximum number of episodes to convert")
    parser.add_argument(
        "--episodes-per-shard", type=int, default=100, help="Number of episodes per TFRecord shard"
    )
    parser.add_argument(
        "--compression",
        choices=["GZIP", "ZLIB", "NONE"],
        default="GZIP",
        help="Compression codec for TFRecord shards",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = ConverterConfig.from_args(args)
    counters = convert_dataset(config)

    print(
        "Conversion complete: {episodes} episodes, {steps} steps across {shards} shard(s).".format(
            **counters
        )
    )


if __name__ == "__main__":
    main()
