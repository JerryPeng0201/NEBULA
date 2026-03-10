from copy import deepcopy

import numpy as np
import sapien
import torch

from nebula import PACKAGE_ASSET_DIR
from nebula.core.embodiment.manipulator import BaseAgent, Keyframe
from nebula.core.embodiment.actuators import *
from nebula.core.embodiment.registration import register_agent
from nebula.utils import common, sapien_utils
from nebula.utils.structs.actor import Actor


@register_agent()
class XArm6GripperG2(BaseAgent):
    uid = "xarm6_gripper_g2"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/xarm6/xarm6_with_gripper.urdf"

    urdf_config = dict(
        _materials=dict(
            gripper=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
        ),
        link=dict(
            left_finger=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
            right_finger=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
        ),
    )

    keyframes = dict(
        rest=Keyframe(
            qpos=np.array(
                [
                    0,
                    0.22,
                    -1.23,
                    0,
                    1.01,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ]
            ),
            pose=sapien.Pose([0, 0, 0]),
        ),
        zeros=Keyframe(
            qpos=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            pose=sapien.Pose([0, 0, 0]),
        ),
    )

    arm_joint_names = [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
    ]

    arm_stiffness = 1e4
    arm_damping = 1e3
    arm_friction = 0.1
    arm_force_limit = 100

    gripper_stiffness = 1e5
    gripper_damping = 2000
    gripper_force_limit = 0.1
    gripper_friction = 1
    ee_link_name = "eef"

    @property
    def _controller_configs(self):
        # -------------------------------------------------------------------------- #
        # Arm
        # -------------------------------------------------------------------------- #
        arm_pd_joint_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=None,
            upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            friction=self.arm_friction,
            force_limit=self.arm_force_limit,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=-0.1,
            upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            friction=self.arm_friction,
            use_delta=True,
        )
        arm_pd_joint_target_delta_pos = deepcopy(arm_pd_joint_delta_pos)
        arm_pd_joint_target_delta_pos.use_target = True

        # PD ee position
        arm_pd_ee_delta_pos = PDEEPosControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            friction=self.arm_friction,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
        )
        arm_pd_ee_delta_pose = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            rot_lower=-0.1,
            rot_upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            friction=self.arm_friction,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
        )
        arm_pd_ee_pose = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=None,
            pos_upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            friction=self.arm_friction,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
            use_delta=False,
            normalize_action=False,
        )

        arm_pd_ee_target_delta_pos = deepcopy(arm_pd_ee_delta_pos)
        arm_pd_ee_target_delta_pos.use_target = True
        arm_pd_ee_target_delta_pose = deepcopy(arm_pd_ee_delta_pose)
        arm_pd_ee_target_delta_pose.use_target = True

        # PD joint velocity
        arm_pd_joint_vel = PDJointVelControllerConfig(
            self.arm_joint_names,
            -1.0,
            1.0,
            self.arm_damping,  # this might need to be tuned separately
            self.arm_force_limit,
            self.arm_friction,
        )

        # PD joint position and velocity
        arm_pd_joint_pos_vel = PDJointPosVelControllerConfig(
            self.arm_joint_names,
            None,
            None,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            self.arm_friction,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos_vel = PDJointPosVelControllerConfig(
            self.arm_joint_names,
            -0.1,
            0.1,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            friction=self.arm_friction,
            use_delta=True,
        )

        # -------------------------------------------------------------------------- #
        # Gripper
        # -------------------------------------------------------------------------- #

        passive_finger_joint_names = [
            "left_finger_joint",
            "right_finger_joint",
            "left_inner_knuckle_joint",
            "right_inner_knuckle_joint",
        ]

        passive_finger_joints = PassiveActuatorConfig(
            joint_names=passive_finger_joint_names,
            damping=0,
            friction=0,
        )

        finger_joint_names = ["left_outer_knuckle_joint", "right_outer_knuckle_joint"]

        finger_mimic_pd_joint_pos = PDJointPosMimicControllerConfig(
            finger_joint_names,
            lower=None,
            upper=None,
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
            friction=self.gripper_friction,
            normalize_action=False,
        )

        finger_mimic_pd_joint_delta_pos = PDJointPosMimicControllerConfig(
            joint_names=finger_joint_names,
            lower=-0.15,
            upper=0.15,
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
            friction=self.gripper_friction,
            normalize_action=True,
            use_delta=True,
        )

        controller_configs = dict(
            pd_joint_delta_pos=dict(
                arm=arm_pd_joint_delta_pos,
                gripper_active=finger_mimic_pd_joint_delta_pos,
                gripper_passive=passive_finger_joints,
            ),
            pd_joint_pos=dict(
                arm=arm_pd_joint_pos,
                gripper_active=finger_mimic_pd_joint_pos,
                gripper_passive=passive_finger_joints,
            ),
            pd_ee_delta_pos=dict(
                arm=arm_pd_ee_delta_pos,
                gripper_active=finger_mimic_pd_joint_delta_pos,
                gripper_passive=passive_finger_joints,
            ),
            pd_ee_delta_pose=dict(
                arm=arm_pd_ee_delta_pose,
                gripper_active=finger_mimic_pd_joint_pos,
                gripper_passive=passive_finger_joints,
            ),
            pd_ee_pose=dict(
                arm=arm_pd_ee_pose,
                gripper_active=finger_mimic_pd_joint_pos,
                gripper_passive=passive_finger_joints,
            ),
            pd_joint_target_delta_pos=dict(
                arm=arm_pd_joint_target_delta_pos,
                gripper_active=finger_mimic_pd_joint_delta_pos,
                gripper_passive=passive_finger_joints,
            ),
            pd_ee_target_delta_pos=dict(
                arm=arm_pd_ee_target_delta_pos,
                gripper_active=finger_mimic_pd_joint_delta_pos,
                gripper_passive=passive_finger_joints,
            ),
            pd_ee_target_delta_pose=dict(
                arm=arm_pd_ee_target_delta_pose,
                gripper_active=finger_mimic_pd_joint_delta_pos,
                gripper_passive=passive_finger_joints,
            ),
            # Caution to use the following controllers
            pd_joint_vel=dict(
                arm=arm_pd_joint_vel,
                gripper_active=finger_mimic_pd_joint_pos,
                gripper_passive=passive_finger_joints,
            ),
            pd_joint_pos_vel=dict(
                arm=arm_pd_joint_pos_vel,
                gripper_active=finger_mimic_pd_joint_pos,
                gripper_passive=passive_finger_joints,
            ),
            pd_joint_delta_pos_vel=dict(
                arm=arm_pd_joint_delta_pos_vel,
                gripper_active=finger_mimic_pd_joint_delta_pos,
                gripper_passive=passive_finger_joints,
            ),
        )

        # Make a deepcopy in case users modify any config
        return deepcopy_dict(controller_configs)

    def _after_loading_articulation(self):
        # Create drives to enforce four-bar linkage (inner_knuckle <-> finger)
        # Right side
        finger_joint = self.robot.active_joints_map["right_finger_joint"]
        inner_knuckle_joint = self.robot.active_joints_map["right_inner_knuckle_joint"]
        finger_link = finger_joint.get_child_link()
        inner_knuckle_link = inner_knuckle_joint.get_child_link()

        # Connection points computed from URDF geometry to close four-bar linkage
        # p_ik: position on inner_knuckle link (in its local frame)
        # p_f: position on finger link (in its local frame)
        p_ik_right = [0, -0.039, 0.042]
        p_f_right = [0, 0.011465, 0.014961]

        right_drive = self.scene.create_drive(
            inner_knuckle_link, sapien.Pose(p_ik_right),
            finger_link, sapien.Pose(p_f_right),
        )
        right_drive.set_limit_x(0, 0)
        right_drive.set_limit_y(0, 0)
        right_drive.set_limit_z(0, 0)

        # Left side
        finger_joint = self.robot.active_joints_map["left_finger_joint"]
        inner_knuckle_joint = self.robot.active_joints_map["left_inner_knuckle_joint"]
        finger_link = finger_joint.get_child_link()
        inner_knuckle_link = inner_knuckle_joint.get_child_link()

        p_ik_left = [0, 0.039, 0.042]
        p_f_left = [0, -0.011465, 0.014961]

        left_drive = self.scene.create_drive(
            inner_knuckle_link, sapien.Pose(p_ik_left),
            finger_link, sapien.Pose(p_f_left),
        )
        left_drive.set_limit_x(0, 0)
        left_drive.set_limit_y(0, 0)
        left_drive.set_limit_z(0, 0)

        # Disable impossible collisions between gripper-related links
        gripper_links = [
            "right_inner_knuckle",
            "right_outer_knuckle",
            "left_inner_knuckle",
            "left_outer_knuckle",
            "right_finger",
            "left_finger",
            "xarm_gripper_base_link",
            "link5",
        ]
        for link_name in gripper_links:
            link = self.robot.links_map[link_name]
            link.set_collision_group_bit(group=2, bit_idx=31, bit=1)

    def _after_init(self):
        self.finger1_link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "left_finger"
        )
        self.finger2_link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "right_finger"
        )
        self.tcp = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.ee_link_name
        )

    def is_grasping(self, object: Actor, min_force=0.5, max_angle=85):
        l_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger1_link, object
        )
        r_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger2_link, object
        )
        lforce = torch.linalg.norm(l_contact_forces, axis=1)
        rforce = torch.linalg.norm(r_contact_forces, axis=1)

        ldirection = self.finger1_link.pose.to_transformation_matrix()[..., :3, 1]
        rdirection = self.finger2_link.pose.to_transformation_matrix()[..., :3, 1]
        langle = common.compute_angle_between(ldirection, l_contact_forces)
        rangle = common.compute_angle_between(rdirection, r_contact_forces)
        lflag = torch.logical_and(
            lforce >= min_force, torch.rad2deg(langle) <= max_angle
        )
        rflag = torch.logical_and(
            rforce >= min_force, torch.rad2deg(rangle) <= max_angle
        )
        return torch.logical_and(lflag, rflag)

    @staticmethod
    def build_grasp_pose(approaching, closing, center):
        assert np.abs(1 - np.linalg.norm(approaching)) < 1e-3
        assert np.abs(1 - np.linalg.norm(closing)) < 1e-3
        assert np.abs(approaching @ closing) <= 1e-3
        ortho = np.cross(closing, approaching)
        T = np.eye(4)
        T[:3, :3] = np.stack([ortho, closing, approaching], axis=1)
        T[:3, 3] = center
        return sapien.Pose(T)

    def is_static(self, threshold: float = 0.2):
        qvel = self.robot.get_qvel()[..., :-6]
        return torch.max(torch.abs(qvel), 1)[0] <= threshold

    @property
    def tcp_pos(self):
        return self.tcp.pose.p

    @property
    def tcp_pose(self):
        return self.tcp.pose
