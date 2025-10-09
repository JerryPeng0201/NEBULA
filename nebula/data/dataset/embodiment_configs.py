# robot_configs.py
# Configuration file for multiple robot types

ROBOT_CONFIGS = {
    "franka_panda_single_arm_2gripper": {
        "robot_name": "Franka_Emika_Panda",
        "robot_type": "single_arm",
        "limbs": {
            "arm": {
                "dof": 7,
                "joint_names": ["Joint0", "Joint1", "Joint2", "Joint3", "Joint4", "Joint5", "Joint6"]
            },
            "gripper": {
                "dof": 2,
                "joint_names": ["finger_joint0", "finger_joint1"]
            }
        },
        "modality": {
            "state": {
                "arm": {
                    "start": 0,
                    "end": 7,
                    "dataset_key": "obs/agent/qpos"
                },
                "gripper": {
                    "start": 7,
                    "end": 9,
                    "dataset_key": "obs/agent/qpos"
                }
            },
            "action": {
                "arm": {
                    "start": 0,
                    "end": 7,
                    "dataset_key": "actions"
                },
                "gripper": {
                    "start": 7,
                    "end": 8,
                    "dataset_key": "actions"
                }
            },
        },
        "views": {
            "base_camera": {
                "dataset_key": "obs/sensor_data/base_camera/rgb"
            },
            "hand_camera": {
                "dataset_key": "obs/sensor_data/hand_camera/rgb"
            }
        },
    },

    "franka_panda_single_arm_1gripper": {
        "robot_name": "Franka_Emika_Panda",
        "robot_type": "single_arm",
        "limbs": {
            "arm": {
                "dof": 7,
                "joint_names": ["Joint0", "Joint1", "Joint2", "Joint3", "Joint4", "Joint5", "Joint6"]
            },
            "gripper": {
                "dof": 1,
                "joint_names": ["finger_joint"]
            }
        },
        "modality": {
            "state": {
                "arm": {
                    "start": 0,
                    "end": 7,
                    "dataset_key": "obs/agent/qpos"
                },
                "gripper": {
                    "start": 7,
                    "end": 8,
                    "dataset_key": "obs/agent/qpos"
                }
            },
            "action": {
                "arm": {
                    "start": 0,
                    "end": 7,
                    "dataset_key": "actions"
                },
                "gripper": {
                    "start": 7,
                    "end": 8,
                    "dataset_key": "actions"
                }
            },
        },
        "views": {
            "base_camera": {
                "dataset_key": "obs/sensor_data/base_camera/rgb"
            },
            "hand_camera": {
                "dataset_key": "obs/sensor_data/hand_camera/rgb"
            }
        },
    },
}