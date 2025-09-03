"""
This script is adapted from
https://github.com/Mehooz/vision4leg/blob/master/vision4leg/envs/locomotion_gym_env.py
"""

import collections
import copy

import numpy as np
import pybullet
import pybullet_utils.bullet_client as bullet_client
import os
import gymnasium

from gym_env.quad_gym.env.env_config import AllControlModes, ROBOT_BODY_MASS
from gym_env.quad_gym.env.sensors import utils
from gymnasium import spaces
from gym_env.quad_gym.env.robots import robot_config
from gym_env.quad_gym.env.robots import a1
from gym_env.quad_gym.env.randomizer.a1_randomizer_terrain import QUADRUPED_INIT_POSITION


class LocomotionGymEnv(gymnasium.Env):
    """The gym environment for the locomotion tasks."""
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 100
    }

    def __init__(self, gym_params, task, sensors, env_randomizers):
        """Initializes the locomotion gym environment.

        Args:
        Raises:
          ValueError: If the num_action_repeat is less than 1.

        """
        self._gym_params = gym_params
        self._sensors = sensors

        if self._gym_params.RobotParams.robot_class == "A1":
            # We currently only use A1 robot
            self._robot_class = a1.A1
        else:
            raise ValueError("Unknown robot class: {}".format(
                self._gym_params.RobotParams.robot_class))

        self._task = task
        self.fric_coeff = gym_params.SceneParams.friction_coefficient

        if self._gym_params.SimParams.enable_rendering:
            if self._gym_params.SimParams.sim_record_video:
                self._pybullet_client = pybullet
                self._pybullet_client.connect(
                    self._pybullet_client.GUI, options="--width=1280 --height=720 --mp4=\"test.mp4\" --mp4fps=100")
                self._pybullet_client.configureDebugVisualizer(
                    self._pybullet_client.COV_ENABLE_SINGLE_STEP_RENDERING, 1)

            else:
                self._pybullet_client = bullet_client.BulletClient(
                    connection_mode=pybullet.GUI)
                pybullet.configureDebugVisualizer(
                    pybullet.COV_ENABLE_GUI,
                    self._gym_params.SimParams.enable_rendering_gui)
        else:
            self._pybullet_client = bullet_client.BulletClient(
                connection_mode=pybullet.DIRECT)

        self._pybullet_client.setAdditionalSearchPath(
            os.path.join(os.path.dirname(__file__), 'assets'))

        if self._gym_params.SimParams.egl_rendering:
            self._pybullet_client.loadPlugin('eglRendererPlugin')

        self._env_randomizers = env_randomizers if env_randomizers is not None else []
        self._control_mode = AllControlModes[self._gym_params.RobotParams.motor_control_mode]

        self._robot = None
        self._world_dict = {}  # here we can add target poses etc
        self._build_action_space()
        self._hard_reset = True  # reset and create all from beginning when starts
        self.reset()
        self._hard_reset = self._gym_params.SimParams.enable_hard_reset
        self._env_time_step = self._gym_params.SimParams.env_time_step
        self.observation_space = (utils.convert_sensors_to_gym_space_dictionary(self.all_sensors()))

        self._last_frame_time = 0.0
        self._show_reference_id = -1
        self._env_step_counter = 0
        self._last_action = None

    def close(self):
        if hasattr(self, '_robot') and self._robot:
            self._robot.Terminate()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        """Resets the robot's position in the world or rebuild the sim world.

        The simulation world will be rebuilt if self._hard_reset is True.

        Args:
          initial_motor_angles: A list of Floats. The desired joint angles after
            reset. If None, the robot will use its built-in value.
          reset_duration: Float. The time (in seconds) needed to rotate all motors
            to the desired initial values.
          reset_visualization_camera: Whether to reset debug visualization camera on
            reset.

        Returns:
          A numpy array contains the initial observation after reset.
        """

        initial_motor_angles = None
        reset_duration = -1
        reset_visualization_camera = True

        if self._gym_params.SimParams.enable_rendering:
            self._pybullet_client.configureDebugVisualizer(
                self._pybullet_client.COV_ENABLE_RENDERING, 0)

        # Clear the simulation world and rebuild the robot interface. This is a default setting when start the
        # simulation at the first time

        if self._hard_reset:
            #print("---------------------------------")
            self._pybullet_client.resetSimulation()
            self._pybullet_client.setPhysicsEngineParameter(
                numSolverIterations=self._gym_params.SimParams.num_bullet_solver_iterations)
            self._pybullet_client.setTimeStep(self._gym_params.SimParams.sim_time_step_s)
            self._pybullet_client.setGravity(0, 0, -9.8)

            # Rebuild the world.
            self._world_dict = {
                "ground": self._pybullet_client.loadURDF("plane_implicit.urdf")
            }

            self._pybullet_client.changeDynamics(
                self._world_dict["ground"], -1,
                lateralFriction=self._gym_params.SceneParams.friction_coefficient[0],
                spinningFriction=self._gym_params.SceneParams.friction_coefficient[1],
                rollingFriction=self._gym_params.SceneParams.friction_coefficient[2]
            )
            # Rebuild the robot
            self._robot = self._robot_class(
                pybullet_client=self._pybullet_client,
                sensors=self._sensors,
                on_rack=self._gym_params.SimParams.robot_on_rack,
                action_repeat=self._gym_params.SimParams.num_action_repeat,
                motor_control_mode=self._control_mode,
                reset_time=self._gym_params.SimParams.reset_time,
                enable_clip_motor_commands=self._gym_params.SimParams.enable_clip_motor_commands,
                enable_action_filter=self._gym_params.SimParams.enable_action_filter,
                enable_action_interpolation=self._gym_params.SimParams.enable_action_interpolation,
                allow_knee_contact=self._gym_params.SimParams.allow_knee_contact,
                is_render=self._gym_params.SimParams.enable_rendering,
                init_pos=QUADRUPED_INIT_POSITION[self._gym_params.SceneParams.terrain_type])

        self._robot.Reset(reload_urdf=False, default_motor_angles=initial_motor_angles, reset_time=reset_duration)

        if self._gym_params.RobotParams.robot_mass_rescale is not None:
            rescaled_mass = self._gym_params.RobotParams.robot_mass_rescale * ROBOT_BODY_MASS
            self._pybullet_client.changeDynamics(self._robot.quadruped, -1, mass=rescaled_mass)

        self._pybullet_client.setPhysicsEngineParameter(enableConeFriction=0)
        self._env_step_counter = 0

        if reset_visualization_camera:
            self._pybullet_client.resetDebugVisualizerCamera(self._gym_params.SimParams.camera_distance,
                                                             self._gym_params.SimParams.camera_yaw,
                                                             self._gym_params.SimParams.camera_pitch,
                                                             [0, 0, 0])

        self._last_action = np.zeros(self.action_space.shape)

        for s in self.all_sensors():
            s.on_reset(self)

        if self._task and hasattr(self._task, 'reset'):
            self._task.reset(self)

        for env_randomizer in self._env_randomizers:
            env_randomizer.randomize_env(self)

        reset_info = None

        return self._get_observation(), reset_info

    def step(self, action):
        """Step forward the simulation, given the action.

        Args:
          action: Can be a list of desired motor angles for all motors when the
            robot is in position control mode; A list of desired motor torques. Or a
            list of tuples (q, qdot, kp, kd, tau) for hybrid control mode. The
            action must be compatible with the robot's motor control mode. Also, we
            are not going to use the leg space (swing/extension) definition at the
            gym level, since they are specific to Minitaur.

        Returns:
          observations: The observation dictionary. The keys are the sensor names
            and the values are the sensor readings.
          reward: The reward for the current state-action pair.
          done: Whether the episode has ended.
          info: A dictionary that stores diagnostic information.

        Raises:
          ValueError: The action dimension is not the same as the number of motors.
          ValueError: The magnitude of actions is out of bounds.
        """

        self._last_base_position = self._robot.GetBasePosition()
        self._last_action = action

        if self._gym_params.SimParams.enable_rendering:
            # Sleep, otherwise the computation takes less time than real time,
            # which will make the visualization like a fast-forward video.
            # time_spent = time.time() - self._last_frame_time
            # self._last_frame_time = time.time()
            # time_to_sleep = self._env_time_step - time_spent
            # if time_to_sleep > 0:
            #     time.sleep(time_to_sleep)
            # base_pos = self._robot.GetBasePosition()

            # # Also keep the previous orientation of the camera set by the user.
            # [yaw, pitch,
            #  dist] = self._pybullet_client.getDebugVisualizerCamera()[8:11]
            # self._pybullet_client.resetDebugVisualizerCamera(dist, yaw, pitch,
            #                                                  base_pos)
            self._pybullet_client.configureDebugVisualizer(
                self._pybullet_client.COV_ENABLE_RENDERING, 1)

        for env_randomizer in self._env_randomizers:
            env_randomizer.randomize_step(self)

        # robot class and put the logics here.
        self._robot.Step(action)

        for s in self.all_sensors():
            s.on_step(self)

        if self._task and hasattr(self._task, 'update'):
            self._task.update(self)

        reward = self._reward()

        terminated, _ = self._terminated_truncated()

        self._env_step_counter += 1

        return self._get_observation(), reward, terminated, _, {}

    def _terminated_truncated(self):
        terminated = False

        if not self._robot.is_safe:
            print("robot not safe")
            terminated = True

        if self._task and hasattr(self._task, 'done'):
            terminated = self._task.done(self)

        for s in self.all_sensors():
            s.on_terminate(self)

        truncated = None  # we determine the truncation in the outerloop
        # truncated = self._env_step_counter >= self._gym_params.TaskParams.max_episode_steps

        return terminated, truncated

    def _reward(self):
        if self._task:
            return self._task(self)
        return 0

    def _get_observation(self):
        """Get observation of this environment from a list of sensors.

        Returns:
          observations: sensory observation in the numpy array format
        """

        sensors_dict = {}
        for s in self.all_sensors():
            sensors_dict[s.get_name()] = s.get_observation()

        for r in self._env_randomizers:
            if hasattr(r, 'env_info'):
                sensors_dict[r.get_name()] = r.env_info

        observations = collections.OrderedDict(sorted(list(sensors_dict.items())))

        return observations

    def get_vision_observation(self, return_label=False):

        rot_pos, rot_orn = self.pybullet_client.getBasePositionAndOrientation(self._robot.quadruped)

        width = 240
        height = 240
        fov = 90
        near_val = 0.1
        far_val = 5

        aspect = width / height
        proj_mat = self.pybullet_client.computeProjectionMatrixFOV(fov,
                                                                   aspect,
                                                                   near_val,
                                                                   far_val)

        rot_mat = self.pybullet_client.getMatrixFromQuaternion(rot_orn)
        forward_vec = [rot_mat[0], rot_mat[3], rot_mat[6]]
        cam_pos = [rot_pos[i] + forward_vec[i] * 0.239 for i in range(3)]
        cam_orn = copy.deepcopy(rot_orn)
        forward_vec2 = [rot_mat[0], rot_mat[3], rot_mat[6]]
        cam_up_vec = [rot_mat[2], rot_mat[5], rot_mat[8]]

        cam_target = [cam_pos[i] + forward_vec2[i] * 10 for i in range(3)]

        view_mat2 = self.pybullet_client.computeViewMatrix(cam_pos, cam_target, cam_up_vec)

        camera_image_set = self.pybullet_client.getCameraImage(
            width, height, viewMatrix=view_mat2, projectionMatrix=proj_mat,
            shadow=1,
            lightDirection=[1, 1, 1],
            renderer=pybullet.ER_BULLET_HARDWARE_OPENGL,
            flags=self.pybullet_client.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX
        )

        imgW, imgH, rgb, depth, seg = camera_image_set

        if return_label:
            info = {"cam_pos": cam_pos, "cam_orn": cam_orn,
                    "rot_pos": rot_pos, "rot_orn": rot_orn, "view_matrix": view_mat2,
                    "projection_matrix": proj_mat,
                    "width": imgW,
                    "height": imgH}
        else:
            info = None

        return rgb, depth, seg, info

    def _build_action_space(self):
        """Builds action space based on motor control mode."""
        motor_mode = AllControlModes[self._gym_params.RobotParams.motor_control_mode]

        if motor_mode == robot_config.MotorControlMode.HYBRID:
            action_upper_bound = []
            action_lower_bound = []
            action_config = self._robot_class.ACTION_CONFIG
            for _ in action_config:
                action_upper_bound.extend([6.28] * 5)
                action_lower_bound.extend([-6.28] * 5)
            self.action_space = spaces.Box(np.array(action_lower_bound), np.array(action_upper_bound), dtype=np.float32)
        elif motor_mode == robot_config.MotorControlMode.TORQUE:
            torque_limits = np.array([100] * len(self._robot_class.ACTION_CONFIG))
            self.action_space = spaces.Box(-torque_limits, torque_limits, dtype=np.float32)
        else:
            # Position mode
            action_upper_bound = []
            action_lower_bound = []
            action_config = self._robot_class.ACTION_CONFIG
            for action in action_config:
                action_upper_bound.append(action.upper_bound)
                action_lower_bound.append(action.lower_bound)
            self.action_space = spaces.Box(np.array(action_lower_bound), np.array(action_upper_bound), dtype=np.float32)

    def render(self, mode='rgb_array'):
        if mode != 'rgb_array':
            raise ValueError('Unsupported render mode:{}'.format(mode))
        base_pos = self._robot.GetBasePosition()
        view_matrix = self._pybullet_client.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=base_pos,
            distance=self._gym_params.SimParams.camera_distance,
            yaw=self._gym_params.SimParams.camera_yaw,
            pitch=self._gym_params.SimParams.camera_pitch,
            roll=0,
            upAxisIndex=2)
        proj_matrix = self._pybullet_client.computeProjectionMatrixFOV(
            fov=60,
            aspect=float(self._gym_params.SimParams.camera_width) / self._gym_params.SimParams.camera_height,
            nearVal=0.1,
            farVal=100.0)
        (_, _, px, _, _) = self._pybullet_client.getCameraImage(
            width=self._gym_params.SimParams.camera_width,
            height=self._gym_params.SimParams.camera_height,
            renderer=pybullet.ER_BULLET_HARDWARE_OPENGL,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix)
        rgb_array = np.array(px)
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def get_states_robot(self):
        """
        Returns the states of the robot in world_coordinates
        """
        states_dict = {"position": self.robot.GetBasePosition(),
                       "orientation": self._robot.GetTrueBaseRollPitchYaw(),
                       "velocity": self._robot.GetBaseVelocity()}
        return states_dict

    def get_vision_states(self):
        """
        This function is used for returning the vision states (rgbd images) on request
        """
        pass

    def get_dynamics_states(self):
        """
        This function is used for returning the dynamics states (rgbd images) on request
        """
        pass

    @property
    def world_dict(self):
        return self._world_dict

    def all_sensors(self):
        return self._robot.GetAllSensors()

    @property
    def pybullet_client(self):
        return self._pybullet_client

    @property
    def robot(self):
        return self._robot

    @property
    def env_step_counter(self):
        return self._env_step_counter

    @property
    def hard_reset(self):
        return self._hard_reset

    @property
    def last_action(self):
        return self._last_action

    @property
    def env_time_step(self):
        return self._env_time_step

    @property
    def task(self):
        return self._task

    @property
    def robot_class(self):
        return self._robot_class

    @property
    def ground_id(self):
        return self._world_dict['ground']

    @ground_id.setter
    def ground_id(self, id):
        self._world_dict['ground'] = id

    @property
    def get_ground(self):
        """Get simulation ground model."""
        return self._world_dict['ground']
