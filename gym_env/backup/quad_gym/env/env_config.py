import dataclasses
import enum
from dataclasses import field
from typing import List
from gym_env.quad_gym.env.robots import robot_config
from gym_env.quad_gym.env.sensors import robot_sensors, environment_sensors

ROBOT_BODY_MASS = 4.713  # move to the parameter list

AllSensors = {
    # report the last action taken
    "EnvLastActionSensor": environment_sensors.LastActionSensor,
    # report the goal position and the robot position
    "EnvGoalPosSensor": environment_sensors.GoalPosSensor,
    # report the foot contact forces for 4 legs, each leg has [Fx, Fy, Fz, Mx, My, Mz]
    "EnvFeetContactSensor": environment_sensors.ForceSensor,
    # report the motor angles for the robot joints, the return value can be in sin/cos form, with/without noise
    "RobotMotorAngleSensor": robot_sensors.MotorAngleSensor,
    # report the motor velocities for the robot joints, the return value can be normalized, with/without noise
    "RobotMotorVelSensor": robot_sensors.MotorVelSensor,
    # report the pose of the leg, the values are calculated from the motor angles. I don't know how to use this.
    "RobotLegPoseSensor": robot_sensors.MinitaurLegPoseSensor,
    # report the displacement of the base (change of the position of the base, this can be
    # projected to a 2D plane by only considering the yaw angle), Cartesian world coordinates
    "RobotBaseDisplacementSensor": robot_sensors.BaseDisplacementSensor,
    # report the displacement of the base (change of the position and orientation of the base, this can be projected
    # to a 2D plane by only considering the yaw angle), Cartesian world coordinates
    "RobotBaseDisplacementAndRotationSensor": robot_sensors.BaseDisplacementAndRotateSensor,
    # report the orientations and angular velocities of the base, can be noisy or true reading
    "RobotIMUSensor": robot_sensors.IMUSensor,
    # report the position of the base, Cartesian world coordinates
    "RobotPositionSensor": robot_sensors.BasePositionSensor,
    # report the pose of the base, Cartesian world coordinates only X-y position and yaw angle
    "RobotPoseSensor": robot_sensors.PoseSensor,
    # report the pose of the base, Cartesian world coordinates only X-y position and yaw angle
    "RobotBaseVelocitySensor": robot_sensors.BaseVelocitySensor,
}

AllControlModes = {
    "position": robot_config.MotorControlMode.POSITION,
    "torque": robot_config.MotorControlMode.TORQUE,
    "hybrid": robot_config.MotorControlMode.HYBRID
}


class TerrainType(enum.Enum):
    """The randomzied terrain types we can use in the gym env."""
    PLANE = 0
    RANDOM_BLOCKS = 1
    TRIANGLE_MESH = 2
    RANDOM_HEIGHTFIELD = 3
    RANDOM_BLOCKS_SPARSE = 4
    RANDOM_HILL = 5
    RANDOM_MOUNT = 6
    MAZE = 7
    STAIRS = 8
    RANDOM_BLOCKS_SPARSE_AND_HEIGHTFIELD = 9
    GOAL_MOUNT = 10
    RANDOM_BLOCKS_SPARSE_WITH_SUBGOAL = 11
    RANDOM_BLOCKS_SPARSE_WITH_SUBGOAL_HEIGHTFIELD = 12
    RANDOM_SPHERE_WITH_SUBGOAL = 13
    MULTI_STAIRS = 14
    RANDOM_BLOCKS_SPARSE_THIN_WIDE = 15
    RANDOM_CHAIR_DESK = 16
    MULTI_SAFE = 17
    BEV_TEST = 18
    GRASS = 19


TerrainTypeDict = {
    "plane": TerrainType.PLANE,
    "random_blocks": TerrainType.RANDOM_BLOCKS,
    "triangle_mesh": TerrainType.TRIANGLE_MESH,
    "random_heightfield": TerrainType.RANDOM_HEIGHTFIELD,
    "random_blocks_sparse": TerrainType.RANDOM_BLOCKS_SPARSE,
    "random_hill": TerrainType.RANDOM_HILL,
    "random_mount": TerrainType.RANDOM_MOUNT,
    "random_maze": TerrainType.MAZE,
    "stairs": TerrainType.STAIRS,
    "random_blocks_sparse_and_heightfield": TerrainType.RANDOM_BLOCKS_SPARSE_AND_HEIGHTFIELD,
    "mount": TerrainType.GOAL_MOUNT,
    "random_blocks_sparse_with_subgoal": TerrainType.RANDOM_BLOCKS_SPARSE_WITH_SUBGOAL,
    "random_blocks_sparse_with_subgoal_heightfield": TerrainType.RANDOM_BLOCKS_SPARSE_WITH_SUBGOAL_HEIGHTFIELD,
    "random_sphere_with_subgoal": TerrainType.RANDOM_SPHERE_WITH_SUBGOAL,
    "multi_stairs": TerrainType.MULTI_STAIRS,
    "random_blocks_sparse_thin_wide": TerrainType.RANDOM_BLOCKS_SPARSE_THIN_WIDE,
    "random_chair_desk": TerrainType.RANDOM_CHAIR_DESK,
    "multi_safe": TerrainType.MULTI_SAFE,
    "bev_test": TerrainType.BEV_TEST,
    "grass": TerrainType.GRASS
}


@dataclasses.dataclass
class SceneConfig:
    env_sensors_list: List[str] = field(default=("EnvLastActionSensor",))
    # The friction coefficient of the ground, lateral, spinning, and rolling friction
    friction_coefficient: List[float] = field(default=(1.0, 0.01, 0.01))
    terrain_type: str = "plane"
    terrain_randomizer: bool = True
    terrain_random_shape: bool = False
    moving: bool = False
    high_range: float = 0.1


@dataclasses.dataclass
class RobotConfig:
    """ The configuration of the robot."""
    robot_class: str = "A1"
    robot_sensors_list: List[str] = field(default=("RobotMotorAngleSensor", "RobotIMUSensor"))
    enable_vision_sensor: bool = False  # if true, the robot will have a camera and return the image as visual observation
    robot_cam_get_image: bool = False
    robot_mass_rescale: float = 1.0
    motor_control_mode: str = "hybrid"
    controller_type: str = 'mpc'
    controller_clip_num: List[float] = field(default_factory=list)

    def __post_init__(self):
        if self.controller_type == 'mpc':
            self.controller_clip_num = [0.3, 0.4]
        elif self.controller_type == 'drl':
            self.controller_clip_num = [0.05, 0.5, 0.5, 0.05, 0.5, 0.5, 0.05, 0.5, 0.5, 0.05, 0.5, 0.5]
        else:
            # self.controller_clip_num = None
            raise ValueError("Unknown controller type: {}".format(self.controller_type))


_ACTION_EPS = 0.01
_NUM_SIMULATION_ITERATION_STEPS = 300
_LOG_BUFFER_LENGTH = 5000


@dataclasses.dataclass
class SimConfig:
    sim_record_video: bool = False
    sim_random_seed: int = 0
    sim_time_step_s: float = 0.001
    num_action_repeat: int = 2 # 2 for mpc and 33 for drl
    enable_hard_reset: bool = False
    enable_rendering: bool = True
    enable_rendering_gui: bool = True
    robot_on_rack: bool = False
    camera_distance: float = 1.0
    camera_pitch: float = -30
    camera_yaw: float = -90
    camera_width: int = 480
    camera_height: int = 360
    egl_rendering: bool = False
    reset_time: float = -1
    enable_action_filter: bool = False
    enable_action_interpolation: bool = True
    allow_knee_contact: bool = False
    enable_clip_motor_commands: bool = False
    env_time_step: float = field(init=False)
    num_bullet_solver_iterations: int = field(init=False)

    def __post_init__(self):
        self.env_time_step = self.sim_time_step_s * self.num_action_repeat
        self.num_bullet_solver_iterations = int(_NUM_SIMULATION_ITERATION_STEPS / self.num_action_repeat)
