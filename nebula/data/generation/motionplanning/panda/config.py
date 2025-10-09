"""
Motion Planning Solutions Configuration
This file contains the mapping of task names to their corresponding solution functions.
"""

from nebula.data.generation.motionplanning.panda.solutions import *

SOLUTIONS_CONFIG = {
   # ===== CONTROL SOLUTIONS =====
   "Control-PlaceSphere-Easy": ControlEasyPlaceSphereSolution,
   "Control-PushCube-Easy": ControlPushCubeEasySolution,
   "Control-StackCube-Easy": ControlStackCubeEasySolution,
   "Control-PegInsertionSide-Medium": ControlPegInsertionSideMediumSolution,
   "Control-PlaceSphere-Medium": ControlPlaceSphereMediumSolution,
   "Control-StackCube-Medium": ControlStackCubeMediumSolution,
   "Control-PlaceSphere-Hard": ControlPlaceSphereHardSolution,
}

