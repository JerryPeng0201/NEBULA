# xarm6_nogripper and xarm6_robotiq use legacy imports (nebula.agents.*)
# and cannot be imported until they are migrated to nebula.core.embodiment.*
# from .xarm6_nogripper import XArm6NoGripper
# from .xarm6_robotiq import XArm6Robotiq, XArm6RobotiqWristCamera
from .xarm6_gripper_g2 import XArm6GripperG2
