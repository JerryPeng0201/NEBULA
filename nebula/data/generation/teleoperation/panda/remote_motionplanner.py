import numpy as np
import grpc
import sapien.core as sapien
import sys
import os
from nebula.core.simulation.engine import BaseEnv

from nebula.utils.structs.pose import to_sapien_pose
import re
from nebula.data.generation.teleoperation.panda import mp_pb2
from nebula.data.generation.teleoperation.panda import mp_pb2_grpc_nebula as mp_pb2_grpc
from pathlib import Path
OPEN = 1
CLOSED = -1

def _pose_to_pq(pose: sapien.Pose):
    pose = to_sapien_pose(pose)
    p, q = pose.p, pose.q
    return [float(p[0]), float(p[1]), float(p[2]), float(q[0]), float(q[1]), float(q[2]), float(q[3])]

class RemotePandaArmMotionPlanningSolver:
    def __init__(self, env: BaseEnv, debug=False, vis=True, base_pose: sapien.Pose = None,
                 visualize_target_grasp_pose=True, print_env_info=True,
                 joint_vel_limits=0.9, joint_acc_limits=0.9,
                 grpc_addr: str = "localhost:50051",
                 urdf_path_override: str | None = None,
                 srdf_path_override: str | None = None):
        self.env = env
        self.base_env: BaseEnv = env.unwrapped
        self.env_agent = self.base_env.agent
        self.robot = self.env_agent.robot
        self.control_mode = self.base_env.control_mode

        self.debug = debug
        self.vis = vis
        self.print_env_info = print_env_info
        self.gripper_state = OPEN
        self.elapsed_steps = 0

        self.channel = grpc.insecure_channel(grpc_addr)
        self.stub = mp_pb2_grpc.PlannerStub(self.channel)

        link_names = [l.get_name() for l in self.robot.get_links()]
        joint_names = [j.get_name() for j in self.robot.get_active_joints()]

        if base_pose is None:
            base_pose = self.env_agent.robot.pose
        base_pose = to_sapien_pose(base_pose)

        urdf_path = urdf_path_override or self.env_agent.urdf_path
        # support in container with /app/assets prefix
        def normalize_urdf_path(urdf_path: str) -> str:
            p = Path(urdf_path).resolve()  
            parts = p.parts
            if "assets" in parts:
                idx = parts.index("assets")
                relative = Path(*parts[idx + 1:])
                new_path = Path("/app/assets") / relative
                return str(new_path)
            else:
                return str(p)
            
        urdf_path = normalize_urdf_path(urdf_path)
        srdf_path = srdf_path_override or urdf_path.replace(".urdf", ".srdf")

    
        self.cfg = mp_pb2.PlannerConfig(
            urdf=urdf_path,
            srdf=srdf_path,
            user_link_names=link_names,
            user_joint_names=joint_names,
            move_group="panda_hand_tcp",
            joint_vel_limits=(np.ones(7) * joint_vel_limits).astype(float).tolist(),
            joint_acc_limits=(np.ones(7) * joint_acc_limits).astype(float).tolist(),
            base_pose_pq=list(np.hstack([base_pose.p, base_pose.q]).astype(float)),
            time_step=float(self.base_env.control_timestep),
        )

    def _reply_to_result(self, rep):
        if rep.status != "Success":
            print(rep.status)
            return -1
        T, dof = rep.traj.n_step, rep.traj.dof
        pos = np.array(rep.traj.position, dtype=np.float64).reshape(T, dof)
        out = {"status": "Success", "position": pos}
        if len(rep.traj.velocity):
            vel = np.array(rep.traj.velocity, dtype=np.float64).reshape(T, dof)
            out["velocity"] = vel
        return out

    def follow_path(self, result, refine_steps: int = 0):
        n_step = result["position"].shape[0]
        for i in range(n_step + refine_steps):
            qpos = result["position"][min(i, n_step - 1)]
            if self.control_mode == "pd_joint_pos_vel" and "velocity" in result:
                qvel = result["velocity"][min(i, n_step - 1)]
                action = np.hstack([qpos, qvel, self.gripper_state])
            else:
                action = np.hstack([qpos, self.gripper_state])
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1
            if self.print_env_info:
                print(f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}")
            if self.vis:
                self.base_env.render_human()
        return obs, reward, terminated, truncated, info

    def move_to_pose_with_RRTConnect(self, pose: sapien.Pose, dry_run: bool = False, refine_steps: int = 0):
        pq = _pose_to_pq(pose)
        qcur = self.robot.get_qpos().cpu().numpy()[0].astype(float).tolist()
        rep = self.stub.PlanToPose(mp_pb2.PlanToPoseRequest(cfg=self.cfg, target_pq=pq, current_qpos=qcur, use_point_cloud=False))
        res = self._reply_to_result(rep)
        if res == -1 or dry_run:
            return res
        return self.follow_path(res, refine_steps=refine_steps)

    def move_to_pose_with_screw(self, pose: sapien.Pose, dry_run: bool = False, refine_steps: int = 0):
        pq = _pose_to_pq(pose)
        qcur = self.robot.get_qpos().cpu().numpy()[0].astype(float).tolist()
        rep = self.stub.PlanScrew(mp_pb2.PlanToPoseRequest(cfg=self.cfg, target_pq=pq, current_qpos=qcur, use_point_cloud=False))
        res = self._reply_to_result(rep)
        if res == -1 or dry_run:
            return res
        return self.follow_path(res, refine_steps=refine_steps)

    def add_collision_pts(self, pts: np.ndarray):
        self.stub.UpdatePointCloud(mp_pb2.UpdatePCRequest(points=pts.astype(np.float32).ravel().tolist(), n=int(pts.shape[0])))

    def clear_collisions(self):
        self.stub.ClearCollisions(mp_pb2.Empty())

    def open_gripper(self):
        self.gripper_state = OPEN
        return self._noop_steps()

    def close_gripper(self, t=6, gripper_state=CLOSED):
        self.gripper_state = gripper_state
        return self._noop_steps(t=t)

    def _noop_steps(self, t: int = 6):
        qpos = self.robot.get_qpos()[0, :-2].cpu().numpy()
        last = None
        for _ in range(t):
            if self.control_mode == "pd_joint_pos":
                action = np.hstack([qpos, self.gripper_state])
            else:
                action = np.hstack([qpos, qpos * 0, self.gripper_state])
            last = self.env.step(action)
            self.elapsed_steps += 1
            if self.print_env_info:
                print(f"[{self.elapsed_steps:3}] Env Output: reward={last[1]} info={last[-1]}")
            if self.vis:
                self.base_env.render_human()
        return last

    def close(self): pass
