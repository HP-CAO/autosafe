import os
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from dataclasses import dataclass, field
tfd = tfp.distributions

@dataclass
class IBConfig:
    batch_size: int = 128
    model_name: str = "IB_model"
    model_path: str = ""

    input_dim: int = 70 # to be overwritten
    embedding_dim: int = 2 # 2 for visualization
    output_dim: int = 4 #to be overwritten
    num_contexts: int = 4 #to be overwritten

    # encoder parameters
    encoder_type: str = "stochastic" # choose from ['deterministic', 'stochastic']
    encoder_hidden_units: list = field(default_factory=lambda: [256, 256, 64])
    encoder_activation: str = "gelu"
    encoder_layer_norm: bool = True
    encoder_learning_rate: float = 0.0003
    encoder_target: bool = False
    soft_update_tau: float = 0.005

    # info maximization parameters
    info_max_type: str = "explicit"  # 'implicit' = InfoNCE, 'explicit' = label reconstruction

    # infoNCE params (used only when info_max_type == "implicit")
    infoNCE_temperature: float = 0.2
    infoNCE_normalization: bool = True  # whether to normalize the embeddings
    infoNCE_num_negatives: int = 3
    infoNCE_sim_type: str = "critic"  # choose from ['critic', 'dot_product']
    critic_learning_rate: float = 0.0005

    # decoder parameters (used only when info_max_type == "explicit")
    decoder_hidden_units: list = field(default_factory=lambda: [32])
    decoder_activation: str = "linear"
    decoder_layer_norm: bool = False
    decoder_learning_rate: float = 0.0005
    decoder_loss_type: str = "cross_entropy"  # choose from ['cross_entropy', 'mse']

    # Variational Information Bottleneck parameters
    beta: float = 0.00001
    beta_bnds: list = field(default_factory=lambda: [0.00001, 0.1])
    beta_schedule: str = "constant"  # choose from ['constant', 'adaptive', 'linear']
    prior: str = "isotropic_gaussian"  # choose from ['isotropic_gaussian', 'ema_gaussian']
    ema_decay: float = 0.99

def generate_tril_dist(params, dim=2):
    """Given flat parameters, assemble a MultivariateTriL Distribution."""
    loc = params[..., :dim]
    tril_params = tfp.bijectors.FillScaleTriL(
        diag_bijector=tfp.bijectors.Exp()).forward(params[..., dim:])
    return tfd.MultivariateNormalTriL(loc=loc, scale_tril=tril_params)

def build_mlp(input_dim, output_dim, hidden_units=[64, 64], activation='gelu', layer_norm=True):
    """Builds a MLP"""
    activation_fn = tf.keras.activations.get(activation)
    layers = []
    in_dim = input_dim
    for h in hidden_units:
        layers.append(tf.keras.layers.Dense(h, input_shape=(in_dim,)))
        if layer_norm:
            layers.append(tf.keras.layers.LayerNormalization())
        layers.append(tf.keras.layers.Activation(activation_fn))
        in_dim = h
    layers.append(tf.keras.layers.Dense(output_dim, input_shape=(in_dim,)))
    return tf.keras.Sequential(layers)

class ContextModel:
    def __init__(self, config: IBConfig):
        """ Variational Information Bottleneck (VIB) Model

        Objective:
        The model optimizes the trade-off between relevance and compression by maximizing:
        I(Z;Y) - beta * I(Z;X),
        where:
        - I(Z;Y) is the mutual information between latent representation Z and target Y.
        - I(Z;X) is the mutual information between latent representation Z and input X.
        - beta is a trade-off parameter.

        Configuration Options:
        
        1. **Encoder Type**:
            - `deterministic`: Performs dimensionality reduction without introducing stochasticity.
            - `stochastic`: Outputs a distribution q(z|x), implemented as a Gaussian with diagonal covariance.

        2. **Information Maximization (I(Z;Y))**:
            - `explicit`: Utilizes a classical lower bound:
                I(Z;Y) >= E[log q(y|z)] + H(Y),
                where q(y|z) is a variational approximation of p(y|z), learned through a decoder.
            - `implicit`: Employs an unnormalized lower bound via a critic-based variational approximation of p(y|z),
                implemented using the InfoNCE loss. Options include:
                - `critic`: Bilinear or dot-product similarity.
                - `contrastive_mode`: Choices are `in-batch` or `explicit_negatives` (for stronger contrastive signals).

        3. **Information Minimization (I(X;Z))**:
            - Enforces a bound using:
                I(X;Z) <= E[KL[q(z|x) || r(z)]],
                where r(z) is a prior distribution. Supported priors include:
                - Isotropic Gaussian.
                - Exponential Moving Average (EMA)-based prior.

        4. **Beta (Trade-off Parameter)**:
            - Governs the balance between maximizing relevance (I(Z;Y)) and minimizing redundancy (I(Z;X)).
            - A monotonically increasing beta schedule is used, transitioning from:
                - Exploitation: Prioritizes stable contextual signals.
                - Exploration: Adds stochasticity to enhance exploration.
        
        This model is designed to jointly optimize with a Contextual Reinforcement Learning (cRL) agent, leveraging the beta schedule to adapt the representation dynamically.
        """
        self.config = config
        
        """=== 1) Build the encoder ==="""
        if self.config.encoder_type == "deterministic":
            self.encoder_net = build_mlp(
                input_dim=config.input_dim,
                output_dim=config.embedding_dim,
                hidden_units=config.encoder_hidden_units,
                activation=config.encoder_activation,
                layer_norm=config.encoder_layer_norm)
        else:
            # Stochastic: output means and scale_tril parameters
            param_dim = config.embedding_dim + config.embedding_dim*(config.embedding_dim+1)//2
            self.encoder_net = build_mlp(
                input_dim=config.input_dim,
                output_dim=param_dim,
                hidden_units=config.encoder_hidden_units,
                activation=config.encoder_activation,
                layer_norm=config.encoder_layer_norm)

        # Optional target encoder for soft updates
        if self.config.encoder_target:
            self.target_encoder_net = build_mlp(
                input_dim=config.input_dim,
                output_dim=param_dim,
                hidden_units=config.encoder_hidden_units,
                activation=config.encoder_activation,
                layer_norm=config.encoder_layer_norm)
            self.target_encoder_net.set_weights(self.encoder_net.get_weights())
        else:
            self.target_encoder_net = None

        self.encoder_optimizer = tf.keras.optimizers.Adam(learning_rate=config.encoder_learning_rate)

        """=== 2) Build the decoder or the critic W for InfoNCE ==="""
        if self.config.info_max_type == "explicit":
            # explicit: reconstruct the labels
            self.decoder_net = build_mlp(
                input_dim=config.embedding_dim,
                output_dim=config.output_dim,
                hidden_units=config.decoder_hidden_units,
                activation=config.decoder_activation,
                layer_norm=config.decoder_layer_norm
            )
            self.decoder_optimizer = tf.keras.optimizers.Adam(learning_rate=config.decoder_learning_rate)
            self.W = None
        else:
            self.decoder_net = None
            self.W_optimizer = tf.keras.optimizers.Adam(learning_rate=config.critic_learning_rate)
            if self.config.infoNCE_sim_type == "critic": # critic-based similarity
                self.W = tf.Variable(initial_value=tf.initializers.GlorotUniform()((config.embedding_dim, config.embedding_dim)),
                trainable=True,
                name="critic_W")
            else:
                self.W = None

        # IB hyperparameters
        self.beta = config.beta
        self.last_info = None

        # Implement EMA-based prior later

    def encode(self, X, use_target=False):
        """Forward pass through the encoder -> z or distribution q(z|X)."""
        if self.config.encoder_target and use_target:
            encoder_out = self.target_encoder_net(X)
        else:
            encoder_out = self.encoder_net(X)

        if self.config.encoder_type == "deterministic":
            return encoder_out  # shape: [B, embedding_dim]
        else:
            dist = generate_tril_dist(encoder_out, dim=self.config.embedding_dim)
            return dist

    def sample_z(self, X, use_target=False, return_type=tf.Tensor):
        """Sample from q(z|X) if stochastic, else just forward pass."""
        qz = self.encode(X, use_target=use_target)
        if self.config.encoder_type == "stochastic":
            z = qz.sample()
        else:
            z = qz
        return z.numpy() if return_type == np.array else z
    
    def sample_z_prior(self, B, return_type=tf.Tensor):
        """Sample from the prior."""
        if self.config.prior == "isotropic_gaussian":
            z = tfd.MultivariateNormalDiag(
                loc=tf.zeros(self.config.embedding_dim),
                scale_diag=tf.ones(self.config.embedding_dim)
            ).sample(B)
        else:
            raise NotImplementedError("EMA-based prior not implemented in this snippet.")
        
        return z.numpy() if return_type == np.array else z

    def update_beta(self, info, progress=None):
        """
        Update the beta parameter for the IB objective.

        Args:
            info: a dict of context->(ave_R, std_R, ave_episode_len, num_samples)
            progress: global step / total steps
        """
        beta_last = self.beta

        if self.last_info is None:
            self.last_info = info
            return beta_last

        if self.config.beta_schedule == "constant": # constant schedule
            self.beta = beta_last
        elif self.config.beta_schedule == "linear": # linear schedule
            if progress is None:
                raise ValueError("Progress must be provided for linear schedule.")
            self.beta = self.config.beta_bnds[0] + progress * (self.config.beta_bnds[1] - self.config.beta_bnds[0])
        elif self.config.beta_schedule == "adaptive":
            ave_R_last = np.mean([v[0] for v in self.last_info.values()]) # last call
            ave_R = np.mean([v[0] for v in info.values()])
            delta_R = ave_R - ave_R_last
            if delta_R < 0:
                pass
        else:
            raise ValueError(f"Unknown beta schedule: {self.config.beta_schedule}")
        
        return beta_last

    def compute_kl_divergence(self, qz):
        """KL[q(z|x) || r(z)], using isotropic Gaussian or an EMA-based prior."""
        if self.config.prior == "isotropic_gaussian":
            pz = tfd.MultivariateNormalDiag(
                loc=tf.zeros(self.config.embedding_dim),
                scale_diag=tf.ones(self.config.embedding_dim)
            )
            return tf.reduce_mean(tfd.kl_divergence(qz, pz))
        elif self.config.prior == "ema_gaussian":
            raise NotImplementedError("EMA-based prior not implemented in this snippet.")
        else:
            return 0.0
    
    def info_loss_implicit(self, anchorZ, posZ, negZ=None):
        """
        Compute the InfoNCE loss.

        Args:
            anchorZ: [B, D]
            posZ: [B, D]
            negZ: [B, N, D] or None
        """
        B = tf.shape(anchorZ)[0]

        # Optional L2 normalization
        if self.config.infoNCE_normalization:
            anchorZ = tf.math.l2_normalize(anchorZ, axis=-1)
            posZ    = tf.math.l2_normalize(posZ,    axis=-1)
            if negZ is not None:
                negZ = tf.math.l2_normalize(negZ, axis=-1)
        
        # Optional critic-based similarity
        if self.W is not None:
            anchorZ = tf.matmul(anchorZ, self.W)  # [B, D]
            posZ    = tf.matmul(posZ,    self.W)  # [B, D]
            if negZ is not None:
                # [B, N, D] -> multiply each of the B*N vectors by W
                negZ = tf.einsum('bnd,dk->bnk', negZ, self.W)  # [B, N, D]

        if negZ is None: # In-batch negatives
            # anchorZ: [B, D], posZ: [B, D]
            sim_matrix = tf.matmul(anchorZ, posZ, transpose_b=True)  # [B, B]
            logits     = sim_matrix / self.config.infoNCE_temperature
            labels     = tf.range(B, dtype=tf.int64)
            loss       = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
            return tf.reduce_mean(loss)
        
        else:
            # Explicit negatives
            # 1) dot(anchor, pos) => [B, 1]
            pos_score = tf.reduce_sum(anchorZ * posZ, axis=-1, keepdims=True)  # [B, 1]

            # 2) dot(anchor, neg) => [B, N]
            neg_score = tf.reduce_sum(
                tf.expand_dims(anchorZ, axis=1) * negZ, axis=-1
            )  # [B, N]

            # 3) concat => [B, 1+N]
            logits = tf.concat([pos_score, neg_score], axis=1)
            logits = logits / self.config.infoNCE_temperature

            # 4) labels = 0
            labels = tf.zeros([B], dtype=tf.int64)
            loss   = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
            return tf.reduce_mean(loss)
        
    def info_loss_explicit(self, Z, labels):
        """ Compute the cross-entropy loss for label reconstruction."""
        logits = self.decoder_net(Z)  # [B, num_contexts]

        if self.config.decoder_loss_type == "mse":
            loss = tf.reduce_mean(tf.keras.losses.MeanSquaredError()(labels, logits))
        else: # use one-hot labels
            loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(labels, logits, from_logits=True))
        return loss
    
    def optimize(self, buffer, num_steps=100, contrastive_mode="in-batch", verbose=False):
        loss_info = {
            "total_loss": [],
            "info_loss": [],
            "kl_loss": []
        }
        for step in range(num_steps):
            if self.config.info_max_type == "explicit":
                X_batch, Y_batch = buffer.sample_by_context(self.config.batch_size)
                X_batch = tf.convert_to_tensor(X_batch, dtype=tf.float32)
                Y_batch = tf.convert_to_tensor(Y_batch)
                total, info, kl = self._train_step(X_batch, Y_batch, mode="explicit")
            else:
                if contrastive_mode == "in-batch":
                    (anchorX, posX), Y_batch = buffer.sample_contrastive(self.config.batch_size, mode="in-batch")
                    anchorX = tf.convert_to_tensor(anchorX, dtype=tf.float32)
                    posX = tf.convert_to_tensor(posX, dtype=tf.float32)
                    total, info, kl = self._train_step(anchorX, Y_batch, posX, mode="implicit_inbatch")
                else:
                    (anchorX, posX, negX), Y_batch = buffer.sample_contrastive(self.config.batch_size, mode="explicit_negatives")
                    anchorX = tf.convert_to_tensor(anchorX, dtype=tf.float32)
                    posX = tf.convert_to_tensor(posX, dtype=tf.float32)
                    negX = tf.convert_to_tensor(negX, dtype=tf.float32)
                    total, info, kl = self._train_step(anchorX, Y_batch, posX, negX, mode="implicit_explicitNeg")
            
            loss_info["total_loss"].append(total.numpy())
            loss_info["info_loss"].append(info.numpy())
            loss_info["kl_loss"].append(kl.numpy())
            
            if verbose and step % 200 == 0:
                print(f"Step: {step}, Total Loss: {total.numpy()}, Info Loss: {info.numpy()}, KL Loss: {kl.numpy()}")
        
        for key in loss_info.keys():
            loss_info[key] = np.mean(loss_info[key])
        return loss_info

    @tf.function
    def _train_step(self, X_batch, Y_batch=None, posX=None, negX=None, mode="explicit"):
        """One training step.

        'mode' can be:
         - "explicit": label reconstruction approach
         - "implicit_inbatch": in-batch InfoNCE
         - "implicit_explicitNeg": explicit negatives InfoNCE
        
        """
        with tf.GradientTape() as tape:
            if self.config.encoder_type == "stochastic":
                qz = self.encode(X_batch)
                z  = qz.sample()
                kl_loss = self.compute_kl_divergence(qz) * self.beta
            else:
                z = self.encode(X_batch)
                kl_loss = 0.0

            if mode == "explicit":
                info_loss = self.info_loss_explicit(z, Y_batch)
            elif mode == "implicit_inbatch":
                posZ = self.sample_z(posX)
                info_loss = self.info_loss_implicit(z, posZ, None)
            else:
                posZ = self.sample_z(posX)

                #flatten the negative samples
                B, k = tf.shape(negX)[0], tf.shape(negX)[1]
                negX_flat = tf.reshape(negX, [B*k, -1])
                negZ_flat = self.sample_z(negX_flat) # [B*k, embedding_dim]
                negZ = tf.reshape(negZ_flat, [B, k, self.config.embedding_dim])

                info_loss = self.info_loss_implicit(z, posZ, negZ)

            total_loss = info_loss + kl_loss


        enc_vars = self.encoder_net.trainable_variables
        if self.decoder_net is not None:
            dec_vars = self.decoder_net.trainable_variables

            grads_all = tape.gradient(total_loss, enc_vars + dec_vars)
            enc_grads = grads_all[:len(enc_vars)]
            dec_grads = grads_all[len(enc_vars):]

            # Apply with separate optimizers
            self.encoder_optimizer.apply_gradients(zip(enc_grads, enc_vars))
            self.decoder_optimizer.apply_gradients(zip(dec_grads, dec_vars))

        elif self.W is not None:
            grads_all = tape.gradient(total_loss, enc_vars + [self.W])
            # Slice out encoder grads vs. W grad
            enc_grads = grads_all[:len(enc_vars)]
            w_grad = grads_all[-1]

            self.encoder_optimizer.apply_gradients(zip(enc_grads, enc_vars))
            self.W_optimizer.apply_gradients([(w_grad, self.W)])
        
        del tape # free up memory

        if self.target_encoder_net is not None and self.config.encoder_target:
            self._soft_update_target()
        
        return total_loss, info_loss, kl_loss

    def _soft_update_target(self):
        """
        Soft update target_encoder_net from encoder_net with factor tau.
        """
        tau = self.config.soft_update_tau
        main_vars   = self.encoder_net.trainable_variables
        target_vars = self.target_encoder_net.trainable_variables
        for mv, tv in zip(main_vars, target_vars):
            tv.assign(tau * mv + (1.0 - tau) * tv)

    def save_model(self, path):
        """Save encoder (and if applicable, decoder) weights."""
        os.makedirs(path, exist_ok=True)

        self.encoder_net.save_weights(os.path.join(path, "encoder_net.weights.h5"))
        if self.decoder_net:
            self.decoder_net.save_weights(os.path.join(path, "decoder_net.weights.h5"))

        if self.target_encoder_net:
            self.target_encoder_net.save_weights(os.path.join(path, "target_encoder_net.weights.h5"))

        # Save critic W if we have one
        if self.W is not None:
            np.save(os.path.join(path, "critic_W.npy"), self.W.numpy())

    def load_model(self, path):
        """Load model weights."""
        self.encoder_net.load_weights(os.path.join(path, "encoder_net.weights.h5"))
        if self.decoder_net:
            self.decoder_net.load_weights(os.path.join(path, "decoder_net.weights.h5"))

        if self.target_encoder_net:
            self.target_encoder_net.load_weights(os.path.join(path, "target_encoder_net.weights.h5"))

        if self.W is not None:
            self.W.assign(np.load(os.path.join(path, "critic_W.npy")))

    def evaluate(self, buffer, num_samples=500):
        """Evaluate prediction accuracy or InfoNCE loss.
        
        Args:
            buffer: ContextualBuffer
            num_samples: number of samples to evaluate on

        Returns:
            if explicit: return prediction accuracy or mean squared error
            if implicit: return InfoNCE loss
        
        """
        if self.config.info_max_type == "explicit":
            X_batch, Y_batch = buffer.sample_by_context(num_samples)
            X_batch = tf.convert_to_tensor(X_batch, dtype=tf.float32)
            Y_batch = tf.convert_to_tensor(Y_batch)
            z = self.sample_z(X_batch)
            logits = self.decoder_net(z)
            if self.config.decoder_loss_type == "cross_entropy":
                acc = tf.reduce_mean(tf.keras.metrics.categorical_accuracy(Y_batch, logits))
                return acc.numpy()
            else:
                mse = tf.reduce_mean(tf.keras.losses.MeanSquaredError()(Y_batch, logits))
                return mse.numpy()
        else:
            (anchorX, posX), Y_batch = buffer.sample_contrastive(num_samples, mode="in-batch")
            anchorX = tf.convert_to_tensor(anchorX, dtype=tf.float32)
            posX = tf.convert_to_tensor(posX, dtype=tf.float32)
            z = self.encode(anchorX)
            posZ = self.encode(posX)
            info_loss = self.info_loss_implicit(z, posZ, None)
            return info_loss.numpy()
        
    def clone(self):
        """ Create a new instance of the model that shares the same weights."""
        cloned_model = ContextModel(self.config)
        cloned_model.encoder_net.set_weights(self.encoder_net.get_weights())

        if self.decoder_net is not None:
            cloned_model.decoder_net.set_weights(self.decoder_net.get_weights())
        
        if self.target_encoder_net is not None:
            cloned_model.target_encoder_net.set_weights(self.target_encoder_net.get_weights())

        if self.W is not None:
            cloned_model.W.assign(self.W.numpy())
        
        return cloned_model
    
    



                

