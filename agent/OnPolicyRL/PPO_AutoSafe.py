import os
import tensorflow as tf
import numpy as np
from dataclasses import dataclass, field

from tensorflow.keras.layers import Input, Dense, LayerNormalization, Lambda
from tensorflow.keras import Model
import tensorflow_probability as tfp

tfd = tfp.distributions

LOG_STD_MAX = 2
LOG_STD_MIN = -20


@tf.function
def get_action_mean(action_mean_learn, action_safe, lam):
    action_mean_learn = tf.math.tanh(action_mean_learn)
    action = (1 - lam) * action_mean_learn + lam * action_safe
    return action

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


def build_critic(shape_input, config: PPOConfig):
    input_layer = Input(shape=(shape_input,), dtype=tf.float32)
    x = build_mlp(input_layer, config.critic_hidden_units, config.critic_activation, config.use_layer_norm)
    value = Dense(1, activation=None)(x)
    model = Model(inputs=input_layer, outputs=value)
    return model


class AutoSafePPO:
    def __init__(self, params: PPOConfig, state_dim, action_dim, P_matrix, F_matrix, mode='train', case='cartpole'):
        self.params = params
        self.state_dim = state_dim
        self.action_dim = action_dim

        if case == 'cartpole':
            from agent.AutoSafe import AutoSafeActor
            self.actor = AutoSafeActor(action_dim, P_matrix, F_matrix, LOG_STD_MAX, LOG_STD_MIN)
        else:
            from agent.AutoSafe import AutoSafeActorQuad
            self.actor = AutoSafeActorQuad(action_dim, P_matrix, F_matrix, LOG_STD_MAX, LOG_STD_MIN)

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
        mean, log_std, action_safe, lam , tem = self.actor(state)

        if deterministic:
            return get_action_mean(mean, action_safe, lam)

        std = tf.exp(log_std)
        normal = tfd.Normal(mean, std)
        action = normal.sample()
        action_drl = tf.tanh(action)
        action = (1 - lam) * action_drl + lam * action_safe
        return action


    @tf.function
    def get_value(self, state):
        return self.critic(tf.expand_dims(state, 0))

    @tf.function
    def get_action_and_log_prob(self, state):
        # [B=1, ...]
        eps = tf.constant(1e-6, dtype=tf.float32)

        state = tf.expand_dims(state, 0)
        mean, log_std, action_safe, lam, tem = self.actor(state)

        std = tf.exp(log_std)
        normal = tfd.Normal(mean, std)

        # sample pre-tanh
        z = normal.sample()
        a_drl = tf.tanh(z)

        action = (1.0 - lam) * a_drl + lam * action_safe
        lam = tf.clip_by_value(lam, 0.0, 1.0 - 1e-3)
        # log prob of the *executed* action via change of variables
        # Gaussian term (sum over dims)
        log_p_z = tf.reduce_sum(normal.log_prob(z), axis=-1)  # shape [B]

        # tanh Jacobian (sum over dims)
        log_det_tanh = tf.reduce_sum(tf.math.log(1.0 - a_drl ** 2 + eps), axis=-1)  # [B]

        # affine Jacobian: handle scalar or per-dim lam generically
        # If lam is [B, D], sum_i log|1-lam_i| ; if [B, 1], this equals d*log|1-lam|
        log_det_affine = tf.reduce_sum(tf.math.log(tf.abs(1.0 - lam) + eps), axis=-1)  # [B]

        old_logp = log_p_z - log_det_tanh - log_det_affine  # [B]
        return tf.squeeze(action, 0), tf.squeeze(old_logp, 0), tf.squeeze(self.critic(state), 0)


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

                    # if np.mean(metrics['kl_div']) > self.params.target_kl * 1.5:
                    #     early_stop = True
                    #     print(f"Early stopping at epoch {epoch} with KL div {np.mean(metrics['kl_div'])}")
                    #     break  # Exit batch loop

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

    # @tf.function
    def _optimize_actor(self, s_batch, a_batch, old_lp_batch, adv_batch):
        # todo stop gradient for old_lp_batch
        # todo reduce epochs
        # with (tf.GradientTape() as tape):
        #     mean, log_std, action_safe, lam, tem = self.actor(s_batch)
        #     std = tf.exp(log_std)
        #     normal = tfd.Normal(mean, std)
        #     a_drl = (a_batch - lam * action_safe) / (1 - lam + 1e-6)
        #     a_drl = tf.clip_by_value(a_drl, -1.0 + 1e-6, 1.0 - 1e-6)
        #     z = tf.atanh(a_drl)
        #     log_probs = tf.reduce_sum(normal.log_prob(z), axis=-1)
        #     log_probs -= tf.reduce_sum(tf.math.log(1 - a_drl ** 2 + 1e-6), axis=-1)
        #
        #     # d = tf.cast(tf.shape(mean)[-1], mean.dtype)
        #     # log_abs_det = d * tf.math.log(tf.abs(1.0 - lam) + 1e-6)
        #     # log_probs = log_probs - log_abs_det
        #
        #     ratio = tf.exp(log_probs - old_lp_batch)
        #     surr1 = ratio * adv_batch
        #     surr2 = tf.clip_by_value(ratio, 1 - self.params.clip_range, 1 + self.params.clip_range) * adv_batch
        #     actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
        #
        #     # Add entropy regularization
        #     entropy = tf.reduce_mean(normal.entropy())
        #     actor_loss -= self.params.entropy_coef * entropy
        #
        #     kl_div_samplewise = old_lp_batch - log_probs
        #     kl_div = tf.reduce_mean(kl_div_samplewise)

        with tf.GradientTape() as tape:
            mean, log_std, action_safe, lam, tem = self.actor(s_batch)  # current policy outputs
            std = tf.exp(log_std)
            normal = tfd.Normal(mean, std)

            eps = 1e-6
            d = tf.cast(tf.shape(mean)[-1], mean.dtype)

            # Invert the current mapping to get a_drl that would produce the *stored* physical action a_batch
            lam = tf.clip_by_value(lam, 0.0, 1.0 - 1e-3)  # keep away from 1
            scale = (1.0 - lam)
            a_drl = (a_batch - lam * action_safe) / (scale + eps)
            a_drl = tf.clip_by_value(a_drl, -1.0 + 1e-6, 1.0 - 1e-6)

            # Squash inverse
            z = tf.atanh(a_drl)

            # Log prob with full change-of-variables: tanh + affine
            log_p_z = tf.reduce_sum(normal.log_prob(z), axis=-1)  # sum over action dims
            log_det_tanh = tf.reduce_sum(tf.math.log(1.0 - a_drl ** 2 + eps), axis=-1)
            log_det_affine = d * tf.math.log(tf.abs(scale) + eps)
            log_probs = log_p_z - log_det_tanh - log_det_affine
            print(log_probs)
            print(old_lp_batch)
            # PPO ratio
            # ratio = tf.exp(tf.clip_by_value(log_probs - old_lp_batch, -20.0, 20.0))
            ratio = tf.exp(log_probs - old_lp_batch)
            print(ratio)
            surr1 = ratio * adv_batch
            surr2 = tf.clip_by_value(ratio, 1 - self.params.clip_range, 1 + self.params.clip_range) * adv_batch
            actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))

            # (Optional) better entropy proxy: include expected Jacobian terms
            entropy_z = tf.reduce_mean(normal.entropy())
            # crude action-entropy proxy
            entropy = entropy_z + tf.reduce_mean(log_det_affine) + tf.reduce_mean(log_det_tanh)
            actor_loss -= self.params.entropy_coef * entropy

            # Sampled KL (old || new): must be consistent with how old_lp_batch was computed
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