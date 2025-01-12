import os
import tensorflow as tf
import numpy as np
from dataclasses import dataclass
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import Model
import tensorflow_probability as tfp


@dataclass
class SACConfig:
    agent_name: str = 'SAC'
    mode: str = 'train'
    soft_alpha: float = 0.005
    learning_rate_actor: float = 0.0003
    learning_rate_critic: float = 0.0003
    batch_size: int = 512
    target_network_frequency: int = 1
    noise_clip: float = 0.5
    entropy_alpha: float = 0.02  
    gamma_discount: float = 0.99
    model_path: str = ''
    total_training_steps: int = 1000000
    replay_buffer_size: int = 1000000
    learning_starts: int = 5000  # this value should be larger than the batch size
    policy_update_frequency: int = 1
    use_layer_norm: bool = True


LOG_STD_MAX = 2
LOG_STD_MIN = -5


def build_actor(shape_input, shape_output, name='', use_layer_norm=False):
    input = Input(shape=(shape_input,), name=name + 'input', dtype=tf.float32)
    if use_layer_norm:
        input = tf.keras.layers.LayerNormalization(axis=-1)(input)
    dense1 = Dense(128, activation='relu', name=name + 'dense1')(input)
    dense2 = Dense(256, activation='relu', name=name + 'dense2')(dense1)
    dense3 = Dense(128, activation='relu', name=name + 'dense3')(dense2)
    dense4 = Dense(64, activation='relu', name=name + 'dense4')(dense3)
    mean = Dense(shape_output, activation=None, name=name + 'mean')(dense4)
    log_std = Dense(shape_output, activation='tanh', name=name + 'std')(dense4)
    # rescaling into interval [LOG_STD_MIN, LOG_STD_MAX] using first order taylor approximation
    log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
    model = Model(inputs=input, outputs=[mean, log_std])
    return model


def build_critic(shape_input, shape_output, name='', use_layer_norm=False):
    input = Input(shape=(shape_input,), name=name + 'input', dtype=tf.float32)
    if use_layer_norm:
        input = tf.keras.layers.LayerNormalization(axis=-1)(input)
    dense1 = Dense(128, activation='relu', name=name + 'dense1')(input)
    dense2 = Dense(256, activation='relu', name=name + 'dense2')(dense1)
    dense3 = Dense(128, activation='relu', name=name + 'dense3')(dense2)
    dense4 = Dense(64, activation='relu', name=name + 'dense4')(dense3)
    q_value = Dense(shape_output, activation=None, name=name + 'q_value')(dense4)
    model = Model(inputs=input, outputs=q_value)
    return model


@tf.function
def get_action_logp(mean, log_std):
    std = tf.math.exp(log_std)  # [bs, n]  in [exp(-5), exp(2)]
    normal = tfp.distributions.Normal(mean, std)  # sample() --> [bs, n]
    eps = tfp.distributions.Normal(loc=tf.zeros_like(std), scale=tf.ones_like(std)).sample()  # [bs, n]
    x_t = mean + eps * std
    action = tf.math.tanh(x_t)
    # Compute log_prob from gaussian, but need to apply correction for tanh squashing.
    # https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/sac/core.py
    log_prob = normal.log_prob(x_t)
    log_prob -= tf.math.log((1 - tf.math.pow(action, 2)) + 1e-5)
    log_prob = tf.reduce_sum(log_prob, axis=-1, keepdims=True)
    return action, log_prob


class SACAgent:
    def __init__(self, params: SACConfig, shape_observations, shape_action, mode='train'):
        self.params = params
        self.actor = None
        self.critic = None
        self.critic_target = None
        self.critic_2 = None
        self.critic_2_target = None

        self.optimizer_actor = tf.keras.optimizers.Adam(learning_rate=self.params.learning_rate_actor)
        self.optimizer_critic =  tf.keras.optimizers.Adam(learning_rate=self.params.learning_rate_critic)
        self.optimizer_critic_2 =  tf.keras.optimizers.Adam(learning_rate=self.params.learning_rate_critic)

        self.shape_action = shape_action
        self.critic_update_step = 0

        if self.params.model_path != '':
            if self.actor is None:
                self.create_model(shape_observations, shape_action)
            self.load_weights(self.params.model_path, mode=mode)
            print("Pretrained model loaded...")
        else:
            self.create_model(shape_observations, shape_action, use_layer_norm=self.params.use_layer_norm)
            self.hard_update()

    def save_weights(self, model_save_path):

        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        self.actor.save_weights(os.path.join(model_save_path, "actor.weights.h5"))
        self.critic.save_weights(os.path.join(model_save_path, "critic.weights.h5"))
        self.critic_target.save_weights(os.path.join(model_save_path, "critic_target.weights.h5"))
        self.critic_2.save_weights(os.path.join(model_save_path, "critic_2.weights.h5"))
        self.critic_2_target.save_weights(os.path.join(model_save_path, "critic_2_target.weights.h5"))

    def load_weights(self, model_path, mode='train'):
        self.actor.load_weights(os.path.join(model_path, "actor.weights.h5"))

        if mode == "train":
            self.critic.load_weights(os.path.join(model_path, "critic.weights.h5"))
            self.critic_target.load_weights(os.path.join(model_path, "critic_target.weights.h5"))
            self.critic_2.load_weights(os.path.join(model_path, "critic_2.weights.h5"))
            self.critic_2_target.load_weights(os.path.join(model_path, "critic_2_target.weights.h5"))

        print("Pretrained weights are loaded")

    def create_model(self, shape_observations, shape_action, use_layer_norm=False):

        self.actor = build_actor(shape_observations, shape_action, name="actor", use_layer_norm=use_layer_norm)
        self.critic = build_critic(shape_observations + shape_action, 1, name="critic", use_layer_norm=use_layer_norm)
        self.critic_target = build_critic(shape_observations + shape_action, 1, name="critic_target", use_layer_norm=use_layer_norm)
        self.critic_2 = build_critic(shape_observations + shape_action, 1, name="critic_2", use_layer_norm=use_layer_norm)
        self.critic_2_target = build_critic(shape_observations + shape_action, 1, name="critic_2_target", use_layer_norm=use_layer_norm)
        self.actor.summary()
        self.critic.summary()

    def hard_update(self):
        self.critic_target.set_weights(self.critic.get_weights())
        self.critic_2_target.set_weights(self.critic_2.get_weights())

    @tf.function
    def soft_update(self):
        soft_alpha = tf.convert_to_tensor(self.params.soft_alpha, dtype=tf.float32)
        self._soft_update(soft_alpha)

    @tf.function
    def _soft_update(self, soft_alpha):
        for w_new, w_old in zip(self.critic.weights, self.critic_target.weights):
            w_old.assign(w_new * soft_alpha + w_old * (1. - soft_alpha))

        for w_new, w_old in zip(self.critic_2.weights, self.critic_2_target.weights):
            w_old.assign(w_new * soft_alpha + w_old * (1. - soft_alpha))

    def get_action(self, observations, mode='train'):
        # this action is for interacting with the environment
        if len(observations.shape) < 2:
            observations = tf.expand_dims(observations, 0)
        mean, log_std = self.actor(observations)
        if mode == 'train':  # Using stochastic policy for exploration
            action, _ = get_action_logp(mean, log_std)
        elif mode == 'test':  # Using deterministic policy for evaluation
            action = tf.math.tanh(mean)
        else:
            raise NotImplementedError(f"{mode} is not implemented")
        return action.numpy()

    def optimize(self, mini_batch):
        
        # check the shape of data first
        for i in range(len(mini_batch)):
            if len(mini_batch[i].shape) == 1:
                mini_batch[i] = np.expand_dims(mini_batch[i], axis=1)

        ob1_tf = tf.convert_to_tensor(mini_batch[0], dtype=tf.float32)
        a1_tf = tf.convert_to_tensor(mini_batch[1], dtype=tf.float32)
        r1_tf = tf.convert_to_tensor(mini_batch[2], dtype=tf.float32)
        ob2_tf = tf.convert_to_tensor(mini_batch[3], dtype=tf.float32)
        ter_tf = tf.convert_to_tensor(mini_batch[4], dtype=tf.float32)
        
        critic_loss, critic_loss_2, q_e, q_e_2 = self._optimize_critic(ob1_tf, a1_tf, r1_tf, ob2_tf, ter_tf)
        self.critic_update_step += 1
        self.soft_update()

        # empirically delayed update make the learning slower
        if self.critic_update_step % self.params.policy_update_frequency == 0:
            actor_loss, log_pi = self._optimize_actor(ob1_tf)
        else:
            actor_loss = 0.0
            log_pi = 0.0

        training_info = {
                         "critic_loss": critic_loss.numpy().mean(),
                         "critic_loss_2": critic_loss_2.numpy().mean(),
                         "q_value": q_e.numpy().mean(),
                         "q_value_2": q_e_2.numpy().mean(),
                         "actor_loss": np.array(actor_loss).mean(),
                         "log_pi": np.array(log_pi).mean()
                         }

        return training_info


    @tf.function
    def _optimize_critic(self, ob1, a1, r1, ob2, ter):

        # ---------------------- optimize critic ----------------------
        mean_next, log_std_next = self.actor(ob2)
        a_next, log_prob_next = get_action_logp(mean_next, log_std_next)
        critic_target_input = tf.concat([ob2, a_next], axis=-1)

        critic_input = tf.concat([ob1, a1], axis=-1)
        q_e = self.critic_target(critic_target_input)  # q value
        q_e_2 = self.critic_2_target(critic_target_input)

        min_q_e = tf.minimum(q_e, q_e_2) - self.params.entropy_alpha * log_prob_next
        y_exp = r1 + self.params.gamma_discount * min_q_e * (1 - ter)

        with tf.GradientTape() as tape:
            y_pre = self.critic(critic_input)
            critic_loss = tf.keras.losses.MeanSquaredError()(y_exp, y_pre)
        q_grads = tape.gradient(critic_loss, self.critic.trainable_variables)

        with tf.GradientTape() as tape2:
            y_pre_2 = self.critic_2(critic_input)
            critic_loss_2 = tf.keras.losses.MeanSquaredError()(y_exp, y_pre_2)
        q_grads_2 = tape2.gradient(critic_loss_2, self.critic_2.trainable_variables)

        self.optimizer_critic.apply_gradients(zip(q_grads, self.critic.trainable_variables))
        self.optimizer_critic_2.apply_gradients(zip(q_grads_2, self.critic_2.trainable_variables))

        return critic_loss, critic_loss_2, q_e, q_e_2,


    @tf.function
    def _optimize_actor(self, ob1):
        with tf.GradientTape() as tape:
            mean, log_std = self.actor(ob1)
            action, log_pi = get_action_logp(mean, log_std)
            critic_input = tf.concat([ob1, action], axis=-1)
            min_q_value = tf.minimum(self.critic(critic_input), self.critic_2(critic_input))
            actor_loss = tf.math.reduce_mean(self.params.entropy_alpha * log_pi - 1 * min_q_value)
        actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.optimizer_actor.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))
        return actor_loss, log_pi

