# isort: off
from .joint_position_control import (
    JointPositionController,
    JointPositionControllerConfig,
    PDJointPosMimicController,
    PDJointPosMimicControllerConfig,
    PDJointPosController,
    PDJointPosControllerConfig,
)

from .cartesian_impedance import (
    CartesianImpedanceController,
    CartesianImpedanceControllerConfig,
    PDEEPosController,
    PDEEPosControllerConfig,
    PDEEPoseController,
    PDEEPoseControllerConfig,
)

from .joint_velocity_control import (
    PDJointVelController, 
    PDJointVelControllerConfig
)

from .joint_pos_vel_control import (
    PDJointPosVelController, 
    PDJointPosVelControllerConfig
)

from .passive_actuator import (
    PassiveActuator, 
    PassiveActuatorConfig
)

from .base_velocity_control import (
    PDBaseVelController, 
    PDBaseVelControllerConfig,
    PDBaseForwardVelController, 
    PDBaseForwardVelControllerConfig
)


def deepcopy_dict(configs: dict):
    """Make a deepcopy of dict.
    The built-in behavior will not copy references to the same value.
    """
    from copy import deepcopy

    assert isinstance(configs, dict), type(configs)
    ret = {}
    for k, v in configs.items():
        if isinstance(v, dict):
            ret[k] = deepcopy_dict(v)
        else:
            ret[k] = deepcopy(v)
    return ret
