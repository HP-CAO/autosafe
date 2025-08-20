import os
import tensorflow as tf
import numpy as np
from dataclasses import dataclass, field
from tensorflow.keras.layers import Input, Dense, LayerNormalization, Lambda
from tensorflow.keras import Model
import tensorflow_probability as tfp

tfd = tfp.distributions

LOG_STD_MIN = -20
LOG_STD_MAX = 2


@dataclass
class PPOConfig:
    agent_name: str = 'PPO'
    mode: str = 'train'
    learning_rate_actor: float = 0.0003
    learning_rate_critic: float = 0.0003
    max_grad_norm_actor: float = 0.5
    max_grad_norm_critic: float = 0.5
    buffer_size: int = 2048
    batch_size: int = 128
    entropy_coef: float = 0.01  # Entropy regularization coefficient
    clip_range: float = 0.2
    gamma_discount: float = 0.99
    gae_lambda: float = 0.97
    ppo_epochs: int = 5
    model_path: str = ''
    total_training_steps: int = 500000
    use_layer_norm: bool = False
    target_kl: float = 0.01

    actor_hidden_units: list = field(default_factory=lambda: [256, 256])
    critic_hidden_units: list = field(default_factory=lambda: [256, 256])
    actor_activation: str = 'relu'
    critic_activation: str = 'relu'


class RolloutBuffer:
    def __init__(self, buffer_size, batch_size, state_dim, action_dim, gamma=0.99, gae_lambda=0.95):
        # Check if buffer size is multiple of batch size
        assert buffer_size % batch_size == 0, \
            f"Buffer size {buffer_size} must be multiple of batch size {batch_size}"
        self.buffer_size = buffer_size
        self.batch_size = batch_size

        self.states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)

        self.ptr = 0
        self.gamma = gamma
        self.gae_lambda = gae_lambda

    def store(self, state, action, reward, done, log_prob, value):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr] = value
        self.ptr += 1

    def compute_advantages_and_returns(self, last_value=0.0):
        advantages = np.zeros(self.ptr, dtype=np.float32)
        last_gae = 0.0
        next_value = last_value

        for t in reversed(range(self.ptr)):
            delta = self.rewards[t] + self.gamma * next_value * (1 - self.dones[t]) - self.values[t]
            last_gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[t]) * last_gae
            advantages[t] = last_gae
            next_value = self.values[t]

        returns = advantages + self.values[:self.ptr]
        return advantages, returns

    def get(self, last_value=0.0):
        assert self.ptr == self.buffer_size, \
            f"Buffer not full! Current: {self.ptr}/{self.buffer_size}"  # Check if buffer is full

        advantages, returns = self.compute_advantages_and_returns(last_value)
        return (
            self.states[:self.ptr],
            self.actions[:self.ptr],
            self.log_probs[:self.ptr],
            advantages,
            returns
        )

    def clear(self):
        self.ptr = 0


def build_mlp(input_layer, hidden_units, activation, use_layer_norm=False):
    x = input_layer
    for units in hidden_units:
        x = Dense(units, activation=activation)(x)
        if use_layer_norm:
            x = LayerNormalization()(x)
    return x


def build_actor(shape_input, shape_output, config: PPOConfig):
    input_layer = Input(shape=(shape_input,), dtype=tf.float32)
    x = build_mlp(input_layer, config.actor_hidden_units, config.actor_activation, config.use_layer_norm)

    # Mean and log_std branches
    mean = Dense(shape_output, activation=None)(x)
    log_std_raw = Dense(shape_output, activation=None)(x)

    # Add clipping as a Keras layer
    log_std = Lambda(lambda x: tf.clip_by_value(x, LOG_STD_MIN, LOG_STD_MAX))(log_std_raw)

    # Create broadcast layer
    class BroadcastLogStd(tf.keras.layers.Layer):
        def call(self, inputs):
            mean_tensor, log_std = inputs
            return tf.broadcast_to(log_std, tf.shape(mean_tensor))

    log_std_broadcast = BroadcastLogStd()([mean, log_std])

    return Model(inputs=input_layer, outputs=[mean, log_std_broadcast])


def build_critic(shape_input, config: PPOConfig):
    input_layer = Input(shape=(shape_input,), dtype=tf.float32)
    x = build_mlp(input_layer, config.critic_hidden_units, config.critic_activation, config.use_layer_norm)
    value = Dense(1, activation=None)(x)
    model = Model(inputs=input_layer, outputs=value)
    return model


class PPOAgent:
    def __init__(self, params: PPOConfig, state_dim, action_dim):
        self.params = params
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.actor = build_actor(state_dim, action_dim, params)
        self.critic = build_critic(state_dim, params)
        self.optimizer_actor = tf.keras.optimizers.Adam(
            learning_rate=params.learning_rate_actor,
            clipnorm=getattr(params, 'max_grad_norm_actor', None)
        )
        self.optimizer_critic = tf.keras.optimizers.Adam(
            learning_rate=params.learning_rate_critic,
            clipnorm=getattr(params, 'max_grad_norm_critic', None)
        )

        self.actor.summary()
        self.critic.summary()

        if params.model_path:
            if os.path.exists(params.model_path):
                self.load_weights(params.model_path)
            else:
                print(f"Warning: Model path {params.model_path} not found. Skipping load.")

        # Warm-up models with a single call
        dummy_state = tf.random.normal((state_dim,))
        self._warmup(dummy_state)

    @tf.function
    def _warmup(self, dummy_state):
        self.get_action(dummy_state)
        self.critic(tf.expand_dims(dummy_state, 0))

    @tf.function
    def get_action(self, state, deterministic=False):
        state = tf.expand_dims(state, 0)
        mean, log_std = self.actor(state)
        log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX)

        if deterministic:
            return tf.tanh(mean)[0]

        std = tf.exp(log_std)
        normal = tfd.Normal(mean, std)
        action = normal.sample()
        return tf.tanh(action)[0]

    @tf.function
    def get_value(self, state):
        return self.critic(tf.expand_dims(state, 0))[0][0]

    @tf.function
    def get_action_and_log_prob(self, state):
        state = tf.expand_dims(state, 0)
        mean, log_std = self.actor(state)
        log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = tf.exp(log_std)

        normal = tfd.Normal(mean, std)
        z = normal.sample()
        action = tf.tanh(z)

        log_prob = normal.log_prob(z)
        log_prob -= tf.reduce_sum(tf.math.log(1 - tf.tanh(z) ** 2 + 1e-6), axis=-1)
        return action[0], log_prob[0], self.critic(state)[0]

    def optimize(self, buffer, last_value=0.0):
        states, actions, old_log_probs, advantages, returns = buffer.get(last_value)
        states_tf = tf.convert_to_tensor(states, dtype=tf.float32)
        actions_tf = tf.convert_to_tensor(actions, dtype=tf.float32)
        old_log_probs_tf = tf.convert_to_tensor(old_log_probs, dtype=tf.float32)
        advantages_tf = tf.convert_to_tensor(advantages, dtype=tf.float32)
        returns_tf = tf.expand_dims(tf.convert_to_tensor(returns, dtype=tf.float32), -1)

        # Normalize advantages
        advantages_tf = (advantages_tf - tf.reduce_mean(advantages_tf)) / (tf.math.reduce_std(advantages_tf) + 1e-8)

        metrics = {'actor_loss': [], 'critic_loss': [], 'entropy': [], 'kl_div': []}
        early_stop = False

        for epoch in range(self.params.ppo_epochs):
            # Create a dataset per epoch with shuffling and batching.
            dataset = tf.data.Dataset.from_tensor_slices(
                (states_tf, actions_tf, old_log_probs_tf, advantages_tf, returns_tf)
            ).shuffle(len(states_tf)).batch(self.params.batch_size).prefetch(tf.data.AUTOTUNE)

            for s_batch, a_batch, old_lp_batch, adv_batch, ret_batch in dataset:
                # Optimize critic
                critic_metrics = self._optimize_critic(s_batch, ret_batch)
                metrics['critic_loss'].append(critic_metrics['critic_loss'])

                # Optimize actor unless early stopping is triggered
                if not early_stop:
                    actor_metrics = self._optimize_actor(s_batch, a_batch, old_lp_batch, adv_batch)
                    metrics['actor_loss'].append(actor_metrics['actor_loss'])
                    metrics['entropy'].append(actor_metrics['entropy'])
                    metrics['kl_div'].append(actor_metrics['kl_div'])

                    if np.mean(metrics['kl_div']) > self.params.target_kl * 1.5:
                        early_stop = True
                        print(f"Early stopping at epoch {epoch} with KL div {np.mean(metrics['kl_div'])}")
                        break  # Exit batch loop

        # Average metrics over all mini-batches
        return {k: np.mean(v) for k, v in metrics.items()}

    @tf.function
    def _optimize_critic(self, s_batch, ret_batch):
        with tf.GradientTape() as tape:
            values = self.critic(s_batch)
            critic_loss = 0.5 * tf.reduce_mean(tf.square(ret_batch - values))

        grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.optimizer_critic.apply_gradients(zip(grads, self.critic.trainable_variables))
        return {'critic_loss': critic_loss}

    @tf.function
    def _optimize_actor(self, s_batch, a_batch, old_lp_batch, adv_batch):
        with tf.GradientTape() as tape:
            mean, log_std = self.actor(s_batch)
            log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX)
            std = tf.exp(log_std)
            normal = tfd.Normal(mean, std)

            a_batch_clipped = tf.clip_by_value(a_batch, -1.0 + 1e-6, 1.0 - 1e-6)
            z = tf.atanh(a_batch_clipped)
            log_probs = tf.reduce_sum(normal.log_prob(z), axis=-1)
            log_probs -= tf.reduce_sum(tf.math.log(1 - a_batch_clipped ** 2 + 1e-6), axis=-1)

            ratio = tf.exp(log_probs - old_lp_batch)
            surr1 = ratio * adv_batch
            surr2 = tf.clip_by_value(ratio, 1 - self.params.clip_range, 1 + self.params.clip_range) * adv_batch
            actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))

            # Add entropy regularization
            entropy = tf.reduce_mean(normal.entropy())
            actor_loss -= self.params.entropy_coef * entropy

            kl_div_samplewise = old_lp_batch - log_probs
            kl_div = tf.reduce_mean(kl_div_samplewise)

        grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.optimizer_actor.apply_gradients(zip(grads, self.actor.trainable_variables))
        return {'actor_loss': actor_loss, 'entropy': entropy, 'kl_div': kl_div}

    def save_weights(self, path):
        self.actor.save_weights(os.path.join(path, 'actor_weights.h5'))
        self.critic.save_weights(os.path.join(path, 'critic_weights.h5'))

    def load_weights(self, path):
        self.actor.load_weights(os.path.join(path, 'actor_weights.h5'))
        self.critic.load_weights(os.path.join(path, 'critic_weights.h5'))