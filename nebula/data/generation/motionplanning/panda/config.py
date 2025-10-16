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
   "Control-PlugCharger-Hard": ControlPlugChargerHardSolution,
   "Control-StackCube-Hard": ControlStackCubeHardSolution,

    # ===== Perception SOLUTIONS =====
    "Perception-PickBiggerSphere-Easy": PerceptionPickBiggerSphereEasySolution,
    "Perception-PickRedSphere-Easy": PerceptionPickRedSphereEasySolution,
    "Perception-PickSphere-Easy": PerceptionPickSphereEasySolution,
    "Perception-PickDiffCubes-Medium": PerceptionPickDiffCubesMediumSolution,
    "Perception-PickRedT-Medium": PerceptionPickRedTMediumSolution,
    "Perception-PickWhitePeg-Medium": PerceptionPickWhitePegMediumSolution,
    "Perception-PickRedT-Hard": PerceptionPickRedTHardSolution,
    "Perception-PickRightCubes-Hard": PerceptionPickRightCubesHardSolution,
    "Perception-PickPeg-Hard": PerceptionPickPegHardSolution,

    # ===== Language SOLUTIONS =====
    "Language-Straight-Easy": LanguageStraightEasySolution,
    "Language-Negation-Medium": LanguageNegationMediumSolution,
    "Language-Condition-Hard": LanguageConditionHardSolution,

    # ===== Dynamic SOLUTIONS =====
    "Dynamic-ColorSwitchPick-Easy": DynamicEasySwitchColorPickSolution,
    "Dynamic-ShapeSwitchPick-Easy": DynamicEasySwitchShapePickSolution,
    "Dynamic-PressSwitch-Easy": DynamicEasyPressSwitchSolution,
    "Dynamic-PickCubeWithCollision-Medium": DynamicMediumCollideSolution,
    "Dynamic-PickCubeWithSliding-Medium": DynamicMediumSlidingPickCubeSolution,
    "Dynamic-PlaceRollingSphere-Medium": DynamicMediumPlaceRollingSphereSolution,
    "Dynamic-DistractorBallPickCube-Hard": DynamicHardDistractorBallPickCubeSolution,
    "Dynamic-CatchRollingSphere-Hard": DynamicHardCatchRollingSphereSolution,
    "Dynamic-RollBall-Hard": DynamicHardRollBallSolution,

    # ===== Spatial SOLUTIONS =====
    "Spatial-MoveCube-Easy": SpatialEasyMoveCubeSolution,
    "Spatial-PickCube-Easy": SpatialEasyPickCubeSolution,
    "Spatial-PlaceBetween-Easy": SpatialEasyPlaceBetweenSolution,
    "Spatial-PickClosest-Medium": SpatialMediumPickClosestSolution,
    "Spatial-PlaceContainer-Medium": SpatialMediumPlaceContainerSolution,
    "Spatial-PickCube-Medium": SpatialMediumPickCubeSolution,
    "Spatial-BuildBlock-Hard": SpatialHardBuildBlockSolution,
    "Spatial-PickCube-Hard": SpatialHardPickCubeSolution,
}

