"""
Motion Planning Solutions Configuration
This file contains the mapping of task names to their corresponding solution functions.
"""

from nebula.data.generation.motionplanning.panda.solutions import *

MP_SOLUTIONS = {
   # ===== CONTROL SOLUTIONS =====
   "Control-PlaceSphere-Easy": ControlEasyPlaceSphereSolution
}

