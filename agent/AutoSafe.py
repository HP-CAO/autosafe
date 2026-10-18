# AutoSafe policy architecture: monitors the safety margin and blends the learned and
# model-based safe actions. It serves as the policy in DRL algorithms such as SAC.
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LayerNormalization, Layer
from tensorflow.keras import Model

class AutoSafeActor(Model):

    def __init__(self, shape_action, P_matrix, F_matrix, log_std_max=2, log_std_min=-5, tem_min=1.0, tem_max=25.0, delta_min=0.0):
        super(AutoSafeActor, self).__init__()

        self.feature_dense_1 = Dense(256, activation='relu', name='feature_dense_1')
        self.feature_dense_2 = Dense(256, activation='relu', name='feature_dense_2')
        self.feature_dense_3 = Dense(256, activation='relu', name='feature_dense_3')

        self.mean_layer = Dense(shape_action, activation=None, name='mean')
        self.std_layer = Dense(shape_action, activation='tanh', name='std')
        self.temperature_layer = Dense(1, activation='tanh', name='temperature') # [0, 1]  # Here temperature can be a learned vector for each action dimension
        self.P_matrix = P_matrix

        # rescale P matrix according to the delta_threshold for safety switching
        # s^t P s < 1 - \delta (\delta is in [0, 1])
        # equivalent to s^t (P * 1 / (1 - \delta)) s < 1
        delta_threshold = min(delta_min, 1.0 - 1e-6) # add a small value to avoid division by zero when delta_threshold is close to 1
        self.P_matrix = self.P_matrix * (1 / (1 - delta_threshold))
        self.F_matrix = F_matrix
        self.n_s, _ = P_matrix.shape
        self.tem_min = tem_min
        self.tem_max = tem_max
        self.log_std_max = log_std_max
        self.log_std_min = log_std_min

    def call(self, inputs):
        input_state = inputs[:, -self.n_s:]  # Assuming the first n_s dimensions are the state
        input_state_expand = tf.expand_dims(input_state, axis=1)
        x = self.feature_dense_1(inputs)
        x = self.feature_dense_2(x)
        x = self.feature_dense_3(x)
        action_mean_learn = self.mean_layer(x)  # Exclude the last dimension for action mean
        tem =self.temperature_layer(x)  # Adding a small value to avoid division by zero #
        tem = self.tem_min + 0.5 * (self.tem_max - self.tem_min) * (tem + 1)  # Rescale temperature to [TEM_MIN, TEM_MAX]
        log_std_learn = self.std_layer(x)
        log_std_learn = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std_learn + 1)
        input_state_transpose = tf.transpose(input_state_expand, perm=[0, 2, 1])  # Transpose for matrix multiplication
        action_safe = tf.clip_by_value(self.F_matrix @ input_state_transpose, -1, 1)  # Clip the action to [-1, 1]
        action_safe = tf.squeeze(action_safe, axis=-1)  # Remove the extra dimension
        safe_z = input_state_expand @ self.P_matrix @ input_state_transpose
        safe_z = tf.clip_by_value(tf.squeeze(safe_z, axis=1), 0 ,1)
        lam = (tf.math.exp(safe_z * tem) - 1) / (tf.math.exp(1 * tem) - 1)
        return action_mean_learn, log_std_learn, action_safe, lam, tem


class AutoSafeActorSchedule(Model):
    def __init__(self, shape_action, P_matrix, F_matrix, log_std_max=2, log_std_min=-5, tem_min=1.0, tem_max=25.0,
                 total_steps=500000, schedule_type="linear", tau=5000):
        super(AutoSafeActorSchedule, self).__init__()

        # networks
        self.feature_dense_1 = Dense(256, activation='relu', name='feature_dense_1')
        self.feature_dense_2 = Dense(256, activation='relu', name='feature_dense_2')
        self.feature_dense_3 = Dense(256, activation='relu', name='feature_dense_3')
        self.mean_layer = Dense(shape_action, activation=None, name='mean')
        self.std_layer = Dense(shape_action, activation='tanh', name='std')

        # safe control
        self.P_matrix = tf.constant(P_matrix, dtype=tf.float32)
        self.F_matrix = tf.constant(F_matrix, dtype=tf.float32)
        self.n_s, _ = P_matrix.shape

        # schedule params
        self.tem_min = tem_min
        self.tem_max = tem_max
        self.total_steps = float(total_steps)
        self.schedule_type = schedule_type
        self.tau = float(tau)

        # std clipping
        self.log_std_max = log_std_max
        self.log_std_min = log_std_min
        self.tem = tf.Variable(initial_value=1.0)  # minimum value is 1.0

    def update_tem(self, step):
        """Monotonically INCREASING temperature from tem_min -> tem_max."""
        step = tf.cast(step, tf.float32)
        t = tf.clip_by_value(step / self.total_steps, 0.0, 1.0)

        if self.schedule_type == "linear":
            # linear up: tem_min  → tem_max
            tem = self.tem_min + (self.tem_max - self.tem_min) * t
        elif self.schedule_type == "exp":
            # exponential up (fast at start, asymptote to tem_max):
            # tem = tem_min + (tem_max - tem_min) * (1 - exp(-step / tau))
            tem = self.tem_min + (self.tem_max - self.tem_min) * (1.0 - tf.exp(-step / self.tau))
        else:
            raise ValueError(f"Unknown schedule_type: {self.schedule_type}")

        self.tem.assign(tem)

    def call(self, inputs):
        """
        inputs: [B, D]
        step: scalar (int or tf.Tensor)
        """
        input_state = inputs[:, -self.n_s:]
        input_state_expand = tf.expand_dims(input_state, axis=1)

        # learned mean/std
        x = self.feature_dense_1(inputs)
        x = self.feature_dense_2(x)
        x = self.feature_dense_3(x)
        action_mean_learn = self.mean_layer(x)

        log_std_learn = self.std_layer(x)
        log_std_learn = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std_learn + 1.0)

        # safe action
        input_state_transpose = tf.transpose(input_state_expand, perm=[0, 2, 1])
        action_safe = tf.clip_by_value(tf.matmul(self.F_matrix, input_state_transpose), -1, 1)
        action_safe = tf.squeeze(action_safe, axis=-1)

        # scheduled temperature
        tem = self.tem

        # λ computation
        safe_z = input_state_expand @ self.P_matrix @ input_state_transpose
        safe_z = tf.clip_by_value(tf.squeeze(safe_z, axis=1), 0, 1)
        lam = tf.clip_by_value((tf.exp(safe_z * tem) - 1.0) / (tf.exp(1.0 * tem) - 1.0), 0.0, 1.0)

        return action_mean_learn, log_std_learn, action_safe, lam, tem


class AutoSafeActorVary(Model):

    def __init__(self, shape_action, P_matrix, F_matrix, log_std_max=2, log_std_min=-5, tem_min=1.0, tem_max=25.0, delta_min=0.0):
        super(AutoSafeActorVary, self).__init__()

        self.feature_dense_1 = Dense(256, activation='relu', name='feature_dense_1')
        self.feature_dense_2 = Dense(256, activation='relu', name='feature_dense_2')
        self.feature_dense_3 = Dense(256, activation='relu', name='feature_dense_3')

        self.mean_layer = Dense(shape_action, activation=None, name='mean')
        self.std_layer = Dense(shape_action, activation='tanh', name='std')
        self.temperature_layer = Dense(1, activation='tanh', name='temperature') # [0, 1]  # Here temperature can be a learned vector for each action dimension
        self.P_matrix = P_matrix

        # Rescale P so that e^T P e >= 1 corresponds to e^T P_orig e >= (1 - delta_min).
        # With delta_min=1 every non-zero error saturates safe_z to 1 → lam=1 (pure safe policy).
        delta_threshold = min(delta_min, 1.0 - 1e-6)
        self.P_matrix = self.P_matrix * (1.0 / (1.0 - delta_threshold))

        self.F_matrix = F_matrix
        self.n_s, _ = P_matrix.shape
        self.tem_min = tem_min
        self.tem_max = tem_max
        self.log_std_max = log_std_max
        self.log_std_min = log_std_min

    def call(self, inputs):
        # obs_aug layout: [obs(n_obs), rescale_ratio(1), safety_tracking_error(n_s), task_tracking_error(n_s)]
        safety_tracking_error = inputs[:, -2 * self.n_s:-self.n_s]  # e_safe = x - x_safe (may be shifted)
        rescaling_ratio       = inputs[:, -2 * self.n_s - 1]        # scalar rescale ratio
        P_matrix = self.P_matrix * tf.reshape(rescaling_ratio, (-1, 1, 1))  # (B,1,1) for broadcasting
        safety_error_expand = tf.expand_dims(safety_tracking_error, axis=1)  # (B,1,n_s)
        safety_error_T = tf.transpose(safety_error_expand, perm=[0, 2, 1])   # (B,n_s,1)
        x = self.feature_dense_1(inputs)
        x = self.feature_dense_2(x)
        x = self.feature_dense_3(x)
        action_mean_learn = self.mean_layer(x)
        tem = self.temperature_layer(x)
        tem = self.tem_min + 0.5 * (self.tem_max - self.tem_min) * (tem + 1)
        log_std_learn = self.std_layer(x)
        log_std_learn = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std_learn + 1)
        action_safe = tf.clip_by_value(self.F_matrix @ safety_error_T, -1, 1)  # safe action w.r.t. safety ref
        action_safe = tf.squeeze(action_safe, axis=-1)
        safe_z = safety_error_expand @ P_matrix @ safety_error_T  # e_safe^T (r*P) e_safe
        safe_z = tf.clip_by_value(tf.squeeze(safe_z, axis=1), 0, 1)
        lam = (tf.math.exp(safe_z * tem) - 1) / (tf.math.exp(1 * tem) - 1)
        return action_mean_learn, log_std_learn, action_safe, lam, tem