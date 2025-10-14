# ubuntu-only file

import numpy as np
import grpc
from concurrent import futures
import logging
import sys, os

import mp_pb2_grpc
sys.path.append(os.path.abspath("."))  # ensure mp_pb2 imports work
import mp_pb2

import mplib  # ubuntu-only
from mplib.pymp import Pose as MPose  # ubuntu-only

# gRPC uses Python's logging system
logging.basicConfig(level=logging.DEBUG)

# Tell gRPC C-core to emit detailed debug logs
os.environ["GRPC_VERBOSITY"] = "DEBUG"
os.environ["GRPC_TRACE"] = "all"



_PLANNER = None
_ALL_COLLISION_PTS = None


def _pq_to_mpose(pq, assume_xyzw=True):
    """pq: [px,py,pz,qx,qy,qz,qw] -> mplib Pose"""
    pq = np.asarray(pq, dtype=np.float64)
    p = pq[:3]
    qx, qy, qz, qw = pq[3:]
    if assume_xyzw:
        q = np.array([qx, qy, qz, qw], dtype=np.float64)  # (x,y,z,w)
    else:
        q = np.array([qw, qx, qy, qz], dtype=np.float64)  # (w,x,y,z) if your build needs it
    return MPose(p=p, q=q)


def _mk_planner(cfg: mp_pb2.PlannerConfig):
    srdf = cfg.srdf if cfg.srdf else cfg.urdf.replace(".urdf", ".srdf")
    planner = mplib.Planner(
        urdf=cfg.urdf,
        srdf=srdf,
        user_link_names=list(cfg.user_link_names),
        user_joint_names=list(cfg.user_joint_names),
        move_group=cfg.move_group or "panda_hand_tcp",
        joint_vel_limits=np.array(cfg.joint_vel_limits, dtype=np.float64),
        joint_acc_limits=np.array(cfg.joint_acc_limits, dtype=np.float64),
    )
    # if len(cfg.base_pose_pq) == 7:
    #     planner.set_base_pose(np.array(cfg.base_pose_pq, dtype=np.float64))
  
    if cfg.base_pose_pq and len(cfg.base_pose_pq) == 7:
        base_pose = _pq_to_mpose(cfg.base_pose_pq, assume_xyzw=True)  # flip to False if needed
        planner.set_base_pose(base_pose)
    return planner

def _ensure_planner(cfg: mp_pb2.PlannerConfig):
    global _PLANNER
    if _PLANNER is None:
        _PLANNER = _mk_planner(cfg)
    return _PLANNER

def _to_reply(res):
    rep = mp_pb2.PlanReply(status=str(res.get("status", "")))
    if rep.status == "Success":
        pos = res["position"]        # (T,7)
        vel = res.get("velocity")    # (T,7) or None
        traj = mp_pb2.Trajectory()
        traj.dof = pos.shape[1]
        traj.n_step = pos.shape[0]
        traj.position.extend(pos.astype(np.float64).ravel().tolist())
        if vel is not None:
            traj.velocity.extend(vel.astype(np.float64).ravel().tolist())
        rep.traj.CopyFrom(traj)
    return rep

def _plan_qpos_to_pose(planner, cfg, target_pq, current_qpos):
    goal_pose = _pq_to_mpose(target_pq, assume_xyzw=True)  # <— convert here
    return planner.plan_qpos_to_pose(
        goal_pose,
        np.asarray(current_qpos, dtype=np.float64),
        time_step=float(cfg.time_step or 0.02),
        # use_point_cloud=bool(getattr(cfg, "use_point_cloud", False)),
        wrt_world=True,
    )

def _plan_screw(planner, cfg, target_pq, current_qpos):
    goal_pose = _pq_to_mpose(target_pq, assume_xyzw=True)  # <— convert here
    return planner.plan_screw(
        goal_pose,
        np.asarray(current_qpos, dtype=np.float64),
        time_step=float(cfg.time_step or 0.02),
        # use_point_cloud=bool(getattr(cfg, "use_point_cloud", False)),
        wrt_world=True,
    )


class PlannerSvc(mp_pb2_grpc.PlannerServicer):
    def PlanToPose(self, req, ctx):
        planner = _ensure_planner(req.cfg)
        res = _plan_qpos_to_pose(planner, req.cfg, req.target_pq, req.current_qpos)
        return _to_reply(res)

    def PlanScrew(self, req, ctx):
        planner = _ensure_planner(req.cfg)
        res = _plan_screw(planner, req.cfg, req.target_pq, req.current_qpos)
        if res.get("status","") != "Success":
            res = _plan_screw(planner, req.cfg, req.target_pq, req.current_qpos)
        return _to_reply(res)

    def UpdatePointCloud(self, req, ctx):
        global _ALL_COLLISION_PTS
        pts = np.array(req.points, dtype=np.float32).reshape(req.n, 3)
        _ALL_COLLISION_PTS = pts if _ALL_COLLISION_PTS is None else np.vstack([_ALL_COLLISION_PTS, pts])
        _ensure_planner(mp_pb2.PlannerConfig())
        _PLANNER.update_point_cloud(_ALL_COLLISION_PTS)
        return mp_pb2.StatusReply(status="OK")

    def ClearCollisions(self, req, ctx):
        global _ALL_COLLISION_PTS
        _ALL_COLLISION_PTS = None
        _ensure_planner(mp_pb2.PlannerConfig())
        _PLANNER.update_point_cloud(np.zeros((0,3), dtype=np.float32))
        return mp_pb2.StatusReply(status="OK")

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    mp_pb2_grpc.add_PlannerServicer_to_server(PlannerSvc(), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    serve()
