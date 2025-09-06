import tensorflow as tf
import numpy as np
from dataclasses import dataclass, field
from tensorflow.keras.layers import Input, Dense, LayerNormalization, Concatenate
from tensorflow.keras import Model

from agent.OffPolicyRL.archived.ReplayBuffer import shape
from agent.model_based.model_based_design_cartpole import MATRIX_P, F

@dataclass
class SACConfig:
    agent_name: str = 'SAC'
    mode: str = 'train'
    soft_alpha: float = 0.005
    learning_rate_actor: float = 0.0003
    learning_rate_critic: float = 0.0003
    batch_size: int = 128
    target_network_frequency: int = 1
    noise_clip: float = 0.5
    entropy_alpha: float = 0.1
    gamma_discount: float = 0.99
    model_path: str = ''
    total_training_steps: int = 500000
    replay_buffer_size: int = 500000
    learning_starts: int = 3000
    policy_update_frequency: int = 1
    use_layer_norm: bool = False

    # Network architecture configurations
    actor_hidden_units: list = field(default_factory=lambda: [256, 256])
    critic_hidden_units: list = field(default_factory=lambda: [256, 256])
    actor_activation: str = 'relu'
    critic_activation: str = 'relu'


LOG_STD_MAX = 2
LOG_STD_MIN = -5


def build_mlp(input_layer, hidden_units, activation, use_layer_norm=False):
    """Builds an MLP given a starting input layer."""
    x = input_layer
    for units in hidden_units:
        x = Dense(units, activation=activation)(x)
        if use_layer_norm:
            x = LayerNormalization()(x)
    return x


def build_actor(shape_input, shape_output, P_matrix, F_matrix, config: SACConfig, name='', ):
    n_s, _ = P_matrix.shape

    input_layer_obs = Input(shape=(shape_input,), name=name+'input', dtype=tf.float32)
    input_layer_state = Input(shape=(n_s,), name=name+'input_state', dtype=tf.float32)

    input_concat = Concatenate()([input_layer_obs, input_layer_state])

    # Build hidden layers
    x = build_mlp(input_concat, config.actor_hidden_units, config.actor_activation, use_layer_norm=config.use_layer_norm)
    mean = Dense(shape_output, activation=None, name=name+'mean')(x)
    log_std = Dense(shape_output, activation='tanh', name=name+'std')(x)
    # rescale log_std to [LOG_STD_MIN, LOG_STD_MAX]
    log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)

    input_transpose_state = tf.transpose(input_layer_state)
    z = input_layer_state @ P_matrix @ input_transpose_state

    model = Model(inputs=[input_layer_obs, input_layer_state], outputs=[mean, log_std, z])
    return model


class AutoSafeActor(Model):

    def __init__(self, shape_action, P_matrix, F_matrix):
        super(AutoSafeActor, self).__init__()

        self.feature_dense_1 = Dense(256, activation='relu', name='feature_dense_1')
        self.feature_dense_2 = Dense(256, activation='relu', name='feature_dense_2')
        self.feature_dense_3 = Dense(256, activation='relu', name='feature_dense_3')

        self.mean_layer = Dense(shape_action, activation=None, name='mean')
        self.std_layer = Dense(shape_action, activation='tanh', name='std')
        self.temperature_layer = Dense(1, activation='sigmoid', name='temperature')
        self.P_matrix = P_matrix
        self.F_matrix = F_matrix
        self.n_s, _ = P_matrix.shape

    def call(self, inputs):
        input_state = inputs[:, :self.n_s]  # Assuming the first n_s dimensions are the state
        input_state_expand = tf.expand_dims(input_state, axis=1)
        x = self.feature_dense_1(inputs)
        x = self.feature_dense_2(x)
        x = self.feature_dense_3(x)
        action_mean_learn = self.mean_layer(x)  # Exclude the last dimension for action mean
        tem = self.temperature_layer(x) + 1e-6  # Adding a small value to avoid division by zero
        log_std_learn = self.std_layer(x)
        log_std_learn = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std_learn + 1)
        input_state_transpose = tf.transpose(input_state_expand, perm=[0, 2, 1])  # Transpose for matrix multiplication
        action_safe = self.F_matrix @ input_state_transpose
        safe_z = input_state_expand @ self.P_matrix @ input_state_transpose
        safe_z = tf.squeeze(safe_z, axis=1)
        lam = tf.clip_by_value((tf.math.exp(safe_z / tem) - 1) / (tf.math.exp(1 / tem) - 1), 0, 1)
        return action_mean_learn, log_std_learn, action_safe, lam


if __name__ == '__main__':
    autosafe_actor = AutoSafeActor(shape_action=1, P_matrix=MATRIX_P, F_matrix=F)
    inputs= tf.constant(np.random.rand(10, 9), dtype=tf.float32)  # Example input for the actor
    outputs = autosafe_actor(inputs)
    print(outputs)
    autosafe_actor.summary()


