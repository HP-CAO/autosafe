import os
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from dataclasses import dataclass, field
tfd = tfp.distributions

@dataclass
class IBConfig:
    batch_size: int = 64 # NOTE: batch size for each context
    learning_rate: float = 0.0003
    model_name: str = "IB_model"
    c_dim: int = 3  # Context dimension (to be overwritten)
    z_dim: int = 5  # Latent dimension
    X_dim: int = 10  # Input feature dimension (to be overwritten)
    encoder_hidden_units: list = field(default_factory=lambda: [128, 256, 256, 128])
    encoder_layer_norm: bool = True
    c_embedding_type: str = "MLP"  # Type of context embedding ('MLP' or 'Linear')
    c_embedding_hidden_units: list = field(default_factory=lambda: [32])
    z_normalization: bool = True
    beta: float = 0.001  # KL-divergence trade-off parameter
    prior: str = "isotropic_gaussian"  # Type of prior
    loss_reduction: str = "mean"  # Loss reduction method ('mean' or 'sum')
    training_period: int = 5000  # Training iterations
    ema_decay: float = 0.99  # Decay rate for EMA
    model_path: str = ""  # Path for model saving/loading

    # used for joint training with RL agent
    freeze_agent_steps: int = 3000  # Number of steps to freeze agent

def generate_tril_dist(params, dim=2):
    """Given flat parameters, assemble a MultivariateTriL Distribution."""
    loc = params[..., :dim]
    tril_params = tfp.bijectors.FillScaleTriL(
        diag_bijector=tfp.bijectors.Exp()).forward(params[..., dim:])
    return tfd.MultivariateNormalTriL(loc=loc, scale_tril=tril_params)

def build_mlp(input_dim, output_dim, hidden_units=[64, 64], activation='gelu', layer_norm=True):
    """Builds a multi-layer perceptron."""
    layers = []
    for units in hidden_units:
        layers.append(tf.keras.layers.Dense(units, activation=activation))
        if layer_norm:
            layers.append(tf.keras.layers.LayerNormalization())
    layers.append(tf.keras.layers.Dense(output_dim))
    return tf.keras.Sequential(layers)

def concat_transitions(
    transitions,
    return_type: str = "numpy",
    minimal: bool = True
):
    """
    Concatenates a list of transitions (s, a, s_next, r, done) into a single 2D array/tensor.
    If 'minimal=True', only [s, a, s_next] are concatenated; else [s, a, s_next, r, done].
    """
    processed = []
    for (s, a, s_next, r, done) in transitions:
        s_arr = np.array(s, dtype=np.float32, copy=False).ravel()
        a_arr = np.array(a, dtype=np.float32, copy=False).ravel()
        sn_arr = np.array(s_next, dtype=np.float32, copy=False).ravel()
        r_arr = np.array([r], dtype=np.float32, copy=False).ravel()
        d_arr = np.array([1.0 if done else 0.0], dtype=np.float32).ravel()

        if minimal:
            row = np.concatenate([s_arr, a_arr, sn_arr], axis=0)
        else:
            # Also include r and done
            row = np.concatenate([s_arr, a_arr, sn_arr, r_arr, d_arr], axis=0)

        processed.append(row)

    data_np = np.stack(processed, axis=0)  # shape [B, D]
    if return_type.lower() == "tensor":
        return tf.convert_to_tensor(data_np, dtype=tf.float32)
    return data_np

class MLPEncoder(tf.keras.layers.Layer):
    """MLP Encoder for dimensionality reduction with optional layer normalization."""
    def __init__(self, input_dim: int, output_dim: int, use_layer_norm: bool, hidden_units: list):
        super(MLPEncoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_layer_norm = use_layer_norm
        self.hidden_units = hidden_units if hidden_units is not None else [64, 64]

        # Build layers
        self.layers_list = self._build_layers()
        self.output_layer = tf.keras.layers.Dense(self.output_dim)

    def _build_layers(self):
        layers = []
        for units in self.hidden_units:
            layers.append(tf.keras.layers.Dense(units, activation='gelu'))
            if self.use_layer_norm:
                layers.append(tf.keras.layers.LayerNormalization())
        return layers

    def call(self, inputs: tf.Tensor, training=False) -> tf.Tensor:
        x = inputs
        for layer in self.layers_list:
            x = layer(x, training=training) if isinstance(layer, tf.keras.layers.LayerNormalization) else layer(x)
        return self.output_layer(x)
    
    def get_config(self):
        config = super(MLPEncoder, self).get_config()
        config.update({
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "use_layer_norm": self.use_layer_norm,
            "hidden_units": self.hidden_units,
        })
        return config


class RNNEncoder(tf.keras.layers.Layer):
    """RNN Encoder for sequential data."""
    def __init__(self, input_dim: int, output_dim: int, encoding_units: list, GRU_units: int):
        super(RNNEncoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.encoding_units = encoding_units if encoding_units is not None else [64, 64]
        self.GRU_units = GRU_units

        # Build layers
        self.embedding_layers = self.build_embedding_layers()
        self.auto_regressive_layer = tf.keras.layers.GRU(self.GRU_units, return_sequences=False)
        self.output_layer = tf.keras.layers.Dense(self.output_dim)

    def build_embedding_layers(self):
        layers = []
        for units in self.encoding_units:
            layers.append(tf.keras.layers.Dense(units, activation='gelu'))
        return layers
    
    def call(self, inputs: tf.Tensor, training=False) -> tf.Tensor:
        x = inputs
        for layer in self.embedding_layers:
            x = layer(x, training=training)
        x = self.auto_regressive_layer(x)
        return self.output_layer(x)
    
    def get_config(self):
        config = super(RNNEncoder, self).get_config()
        config.update({
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "encoding_units": self.encoding_units,
            "GRU_units": self.GRU_units,
        })
        return config
    
# class EmbeddingLayer(tf.keras.layers.Layer):
    """
    Context embedding layer mapping context c to embedding z.
    
    Parameters:
    - input_dim (int): The input dimensionality.
    - output_dim (int): The output dimensionality.
    - embedding_type (str): The type of embedding, e.g., 'MLP'.
    - hidden_units (list of int): The number of units for each hidden layer.
    """
    # def __init__(self, input_dim: int, output_dim: int, embedding_type: str, hidden_units: list = None):
    #     super(EmbeddingLayer, self).__init__()
    #     self.input_dim = input_dim
    #     self.embedding_dim = output_dim
    #     self.embedding_type = embedding_type
    #     self.hidden_units = hidden_units if hidden_units is not None else []
        
    #     # Build layers
    #     self.layers_list = self._build_layers()
    #     self.output_layer = tf.keras.layers.Dense(self.embedding_dim)

    # def _build_layers(self):
    #     layers = []
    #     activation = 'gelu' if self.embedding_type == 'MLP' else None
    #     for unit in self.hidden_units:
    #         layers.append(tf.keras.layers.Dense(unit, activation=activation))
    #     return layers

    # def call(self, c: tf.Tensor) -> tf.Tensor:
    #     x = c
    #     for layer in self.layers_list:
    #         x = layer(x)
    #     return self.output_layer(x)
    
    # def get_config(self):
    #     config = super(EmbeddingLayer, self).get_config()
    #     config.update({
    #         "input_dim": self.input_dim,
    #         "output_dim": self.embedding_dim,
    #         "embedding_type": self.embedding_type,
    #         "hidden_units": self.hidden_units,
    #     })
    #     return config

# class IB_Block(tf.keras.Model):
    """PhyIB Block combining encoder and context embedding layer."""
    # def __init__(self, config: IBConfig):
    #     super(IB_Block, self).__init__()
    #     self.z_dim = config.z_dim
    #     num_params = config.z_dim + (config.z_dim * (config.z_dim + 1)) // 2
    #     self.encoder = MLPEncoder(
    #         input_dim=config.X_dim,
    #         output_dim=num_params,
    #         use_layer_norm=config.encoder_layer_norm,
    #         hidden_units=config.encoder_hidden_units
    #     )
    #     self.embedding_layer = EmbeddingLayer(
    #         input_dim=config.c_dim,
    #         output_dim=config.z_dim,
    #         embedding_type=config.c_embedding_type,
    #         hidden_units=config.c_embedding_hidden_units
    #     )

    #     self.encoder_target = MLPEncoder(
    #         input_dim=config.X_dim,
    #         output_dim=num_params,
    #         use_layer_norm=config.encoder_layer_norm,
    #         hidden_units=config.encoder_hidden_units
    #     ) # note: to be initialized with the same weights as the model encoder
    #     self.encoder_target.set_weights(self.encoder.get_weights())

    # def encode(self, x: tf.Tensor):
    #     params = self.encoder(x)
    #     return generate_tril_dist(params, dim=self.z_dim)

    # def get_target_embedding(self, c: tf.Tensor) -> tf.Tensor:
    #     return self.embedding_layer(c)

    # def call(self, x: tf.Tensor, c: tf.Tensor):
    #     encoded_x = self.encode(x)
    #     return encoded_x

def info_nce_loss(
    queries: tf.Tensor,
    positives: tf.Tensor,
    negatives: tf.Tensor = None,
    contrast_mode: str = "in-batch",
    temperature: float = 0.5,
    reduction: str = 'mean',
    normalize: bool = False
) -> tf.Tensor:
    """
    Computes the InfoNCE loss in two modes:
      1) 'in-batch': standard approach where each row i in `queries` matches row i in `positives`,
         and everything else in `positives` is a negative.
      2) 'explicit_negatives': each row i in `queries` has exactly one positive (row i in `positives`)
         and one or more negatives in `negatives`. We then construct a [batch_size, 1 + K] matrix
         of logits for each anchor, where K is the number of negatives per anchor.

    Args:
        queries:   [B, D] anchor embeddings
        positives: [B, D] positive embeddings (same ordering as queries)
        negatives: 
            - If contrast_mode='in-batch', this can be None.
            - If contrast_mode='explicit_negatives', shape should be [B, K, D]
              (for each of the B anchors, we have K negative samples).
        contrast_mode: 'in-batch' or 'explicit_negatives'
        temperature: Temperature scaling for logits
        reduction: 'mean' or 'sum'
        normalize: Whether to L2-normalize the embeddings before computing similarities.

    Returns:
        A tf.Tensor representing the scalar InfoNCE loss.
    """

    if normalize:
        queries = tf.math.l2_normalize(queries, axis=-1)     # [B, D]
        positives = tf.math.l2_normalize(positives, axis=-1) # [B, D]
        if negatives is not None:
            negatives = tf.math.l2_normalize(negatives, axis=-1)  # shape depends on mode

    if contrast_mode == "in-batch":
        # --------------------------------------------------------------
        # Standard in-batch negatives
        #
        # We compute a [B, B] matrix of logits = queries * positives^T.
        # Row i vs column i is the positive.  Row i vs col j != i is negative.
        # The label for row i is i.
        # --------------------------------------------------------------
        logits = tf.matmul(queries, positives, transpose_b=True)   # shape [B, B]
        logits /= temperature

        labels = tf.range(tf.shape(queries)[0])  # [B]
        loss_vec = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits
        )
        loss = tf.reduce_mean(loss_vec) if reduction == 'mean' else tf.reduce_sum(loss_vec)

    elif contrast_mode == "explicit_negatives":
        # --------------------------------------------------------------
        # Explicit negative samples:
        #   queries:   [B, D]
        #   positives: [B, D]
        #   negatives: [B, K, D], meaning for each of the B anchors,
        #              we have K negative embeddings.
        #
        # We'll build a logit matrix of shape [B, (1 + K)] where:
        #   - The first column is the dot-product of queries[i] with positives[i].
        #   - The next K columns are queries[i] with each of the negatives[i, k].
        #
        # Then the label for each row i is 0, meaning the first column is the correct "positive."
        # --------------------------------------------------------------
        if negatives is None:
            raise ValueError("negatives must be provided for explicit_negatives mode.")

        batch_size = tf.shape(queries)[0]
        # Compute dot with positives:
        pos_logits = tf.reduce_sum(queries * positives, axis=-1, keepdims=True)  # [B, 1]
        # Or equivalently: tf.einsum('bd,bd->b', queries, positives)

        # Compute dot with each negative
        # queries = [B, 1, D], negatives = [B, K, D], so broadcast mul is [B, K, D],
        # then sum over D => shape [B, K]
        queries_ = tf.expand_dims(queries, axis=1)             # => [B, 1, D]
        neg_logits = tf.reduce_sum(queries_ * negatives, axis=-1)  # => [B, K]

        # Concatenate along columns => [B, 1+K]
        all_logits = tf.concat([pos_logits, neg_logits], axis=1)
        all_logits /= temperature

        # Labels: each row i says "column 0" is correct
        labels = tf.zeros(batch_size, dtype=tf.int32)  # i.e., 0 for all
        loss_vec = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=all_logits
        )
        loss = tf.reduce_mean(loss_vec) if reduction == 'mean' else tf.reduce_sum(loss_vec)

    else:
        raise ValueError("Invalid contrast_mode. Use 'in-batch' or 'explicit_negatives'.")

    return loss

# def info_nce_loss(
#     queries: tf.Tensor,
#     positives: tf.Tensor,
#     temperature: float = 0.5,
#     reduction: str = 'mean',
#     normalize: bool = False) -> tf.Tensor:
#     """
#     Computes the InfoNCE loss in in-batch mode.
#     Args:
#         queries: Tensor of shape [batch_size, dim]
#         positives: Tensor of shape [batch_size, dim]
#         temperature: Temperature scaling for logits
#         reduction: 'mean' or 'sum'
#         normalize: Whether to normalize embeddings
#     Returns:
#         tf.Tensor representing the loss
#     """
#     if normalize:
#         # normalize the embeddings
#         queries = tf.math.l2_normalize(queries, axis=-1)
#         positives = tf.math.l2_normalize(positives, axis=-1)

#     logits = tf.matmul(queries, tf.transpose(positives)) / temperature
#     labels = tf.range(tf.shape(queries)[0])
#     loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)

#     return tf.reduce_mean(loss) if reduction == 'mean' else tf.reduce_sum(loss)

# class IBModel:
    # def __init__(self, config: IBConfig):
    #     self.config = config
    #     self.model = IB_Block(config)
    #     self.optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
    #     self.beta = config.beta

    #     if config.prior == 'isotropic_gaussian':
    #         self.prior = tfd.MultivariateNormalDiag(
    #             loc=tf.zeros(config.z_dim),
    #             scale_diag=tf.ones(config.z_dim)
    #         )
    #     else:
    #         self.ema_mu = tf.Variable(tf.zeros(config.z_dim), trainable=False)
    #         self.ema_var = tf.Variable(tf.ones(config.z_dim), trainable=False)
    #         self.ema_decay = config.ema_decay
    #         self.prior = tfd.MultivariateNormalDiag(loc=self.ema_mu, scale_diag=tf.sqrt(self.ema_var))

    #     # initialize target encoder for soft updates during RL training
    #     self.target_encoder = self.model.encoder_target

    # def sample_z(self, X, mode: str = 'posterior', use_target: bool = False, return_type=np.array):
    #     """
    #     Samples z from the specified encoder (target or model).
    #     Args:
    #         X: Input features.
    #         mode: 'posterior', 'prior', or 'posterior_lowvar'.
    #         use_target: Whether to sample from the target encoder (True) or model encoder (False).
    #         return_type: Output type (numpy array or Tensor).
    #     Returns:
    #         Sampled z.
    #     """
    #     encoder = self.target_encoder if use_target else self.model.encoder
    #     params = encoder(X)
    #     z_dist = generate_tril_dist(params, dim=self.config.z_dim)

    #     if mode == 'prior':
    #         z_sample = self.prior.sample()
    #     elif mode == 'posterior':
    #         z_sample = z_dist.sample()
    #     elif mode == 'posterior_lowvar':
    #         z_samples = z_dist.sample(10)
    #         z_sample = tf.reduce_mean(z_samples, axis=0)
    #     else:
    #         raise ValueError("Invalid mode")

    #     return z_sample.numpy() if return_type == np.array else z_sample

    # def optimize(self, data_batch):
    #     x, c = map(tf.convert_to_tensor, data_batch)
    #     total_loss, loss_info, kl_div = self._optimize((x, c))
    #     results = {
    #         "total_loss": total_loss.numpy(),
    #         "infoNCE_loss": loss_info.numpy(),
    #         "rate_loss": kl_div.numpy()
    #     }
    #     # self.soft_update()
    #     return results

    # @tf.function
    # def _optimize(self, data_batch):
    #     x, c = data_batch
    #     with tf.GradientTape() as tape:
    #         z_dist = self.model.encode(x)
    #         z_sample = z_dist.sample()
    #         z_tar = self.model.get_target_embedding(c)

    #         loss_info = info_nce_loss(
    #             queries=z_sample, positives=z_tar, temperature=0.7,
    #             reduction=self.config.loss_reduction, normalize=self.config.z_normalization
    #         )

    #         kl_div = tf.reduce_mean(tfd.kl_divergence(z_dist, self.prior))
    #         total_loss = loss_info + self.beta * kl_div

    #     gradients = tape.gradient(total_loss, self.model.trainable_variables)
    #     self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    #     # Update EMA for aggregated posterior
    #     if self.config.prior != "isotropic_gaussian":
    #         batch_mu = tf.reduce_mean(z_sample, axis=0)
    #         batch_var = tf.math.reduce_variance(z_sample, axis=0)
    #         self.ema_mu.assign(self.ema_decay * self.ema_mu + (1 - self.ema_decay) * batch_mu)
    #         self.ema_var.assign(self.ema_decay * self.ema_var + (1 - self.ema_decay) * batch_var)

    #     return total_loss, loss_info, kl_div

    # @tf.function
    # def soft_update(self, tau=0.005):
    #     """Soft update target encoder."""
    #     for target_param, param in zip(self.target_encoder.trainable_variables, self.model.encoder.trainable_variables):
    #         target_param.assign(tau * param + (1 - tau) * target_param)

    # def save_weights(self, path):
    #     if not os.path.exists(path):
    #         os.makedirs(path)
    #     self.model.save_weights(os.path.join(path, 'ib_model.weights.h5'))
    #     print(f"Model weights saved to {path}")

    # def load_weights(self, path):
    #     self.model.load_weights(path)
    #     print(f"Model weights loaded from {path}")

class IB_Block(tf.keras.Model):
    def __init__(self, config: IBConfig):
        super().__init__()
        self.z_dim = config.z_dim
        num_params = config.z_dim + (config.z_dim*(config.z_dim + 1))//2

        # "online" encoder
        self.encoder = MLPEncoder(
            input_dim=config.X_dim,
            output_dim=num_params,
            use_layer_norm=config.encoder_layer_norm,
            hidden_units=config.encoder_hidden_units
        )

        # "target" encoder for soft-updates
        self.encoder_target = MLPEncoder(
            input_dim=config.X_dim,
            output_dim=num_params,
            use_layer_norm=config.encoder_layer_norm,
            hidden_units=config.encoder_hidden_units
        )

        # Initialize target to have the same weights
        self.encoder_target.set_weights(self.encoder.get_weights())

        # Critic
        self.W = tf.Variable(
            initial_value=tf.initializers.GlorotUniform()((self.z_dim, self.z_dim)),
            trainable=True,
            name="critic_W"
        )

        # predictor
        self.decoder = tf.keras.Dense(config.c_dim)

    def encode(self, x: tf.Tensor, use_target=False):
        if use_target:
            params = self.encoder_target(x)
        else:
            params = self.encoder(x)
  
        loc = params[..., :self.z_dim] # mean
        tril_params = tfp.bijectors.FillScaleTriL(diag_bijector=tfp.bijectors.Exp()).forward(params[..., self.z_dim:]) # lower triangular
        return tfd.MultivariateNormalTriL(loc=loc, scale_tril=tril_params)

    def critic_score(self, z1: tf.Tensor, z2: tf.Tensor) -> tf.Tensor:
        """
        Returns the bilinear form z1^T * W * z2 for each row in z1, z2.
        - z1, z2: shape [B, z_dim]
        - W: shape [z_dim, z_dim]
        Output: shape [B, B] if we do a pairwise score.
        """
        out1 = tf.matmul(z1, self.W)          # [B, z_dim]
        logits = tf.matmul(out1, z2, transpose_b=True)  # => [B, B]
        return logits
    
    def call(self, x, training=False):
        # "call" for a single forward pass (not always used directly in contrastive setups)
        return self.encode(x, use_target=False)


class ContextIBModel:
    def __init__(self, config):
        self.config = config
        self.model = IB_Block(config)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
        self.beta = config.beta

        if config.prior == "isotropic_gaussian":
            self.prior = tfd.MultivariateNormalDiag(
                loc=tf.zeros(config.z_dim),
                scale_diag=tf.ones(config.z_dim)
            )
        else:
            # EMA-based prior
            self.ema_mu = tf.Variable(tf.zeros(config.z_dim), trainable=False)
            self.ema_var = tf.Variable(tf.ones(config.z_dim), trainable=False)
            self.ema_decay = config.ema_decay
            self.prior = tfd.MultivariateNormalDiag(
                loc=self.ema_mu, scale_diag=tf.sqrt(self.ema_var)
            )

    def optimize_in_batch(self, buffer, batch_size,):
        pass


    def optimize_explicit_negatives(self, buffer, neg_k=5, minimal=True):
        x_anchor_list, x_pos_list, x_neg_list = buffer.sample_explicit_negatives(
            batch_size=self.config.batch_size, neg_k=neg_k)

        anchor_arr = concat_transitions(x_anchor_list, return_type="tensor", minimal=minimal)
        pos_arr    = concat_transitions(x_pos_list,    return_type="tensor", minimal=minimal)

        # x_neg_list => shape [B, neg_k], each item is a list of transitions
        # we'll flatten each sub-list separately
        neg_tensors = []
        for neg_sublist in x_neg_list:
            sub_arr = concat_transitions(neg_sublist, return_type="tensor", minimal=minimal)
            neg_tensors.append(sub_arr)  # shape => [neg_k, D]

        # Stack => shape [B, neg_k, D]
        neg_tensor = tf.stack(neg_tensors, axis=0)

        total_loss, info_loss, kl_val = self._train_step_explicit(anchor_arr, pos_arr, neg_tensor)
        return {
            "total_loss": total_loss.numpy(),
            "info_loss": info_loss.numpy(),
            "kl_loss": kl_val.numpy()
        }

    @tf.function
    def _train_step_in_batch(self, x_anchor, x_pos):
        """
        In-batch InfoNCE + IB update:
        - x_anchor, x_pos: shape [B, X_dim]
        """
        temp = 0.5
        B = tf.shape(x_anchor)[0]
        normalize = True

        with tf.GradientTape() as tape:
            z_dist_anchor = self.model.encode(x_anchor, use_target=False)
            z_anchor = z_dist_anchor.sample()

            z_dist_pos = self.model.encode(x_pos, use_target=False)
            z_pos = z_dist_pos.sample()

            if normalize:
                z_anchor = tf.math.l2_normalize(z_anchor, axis=-1)
                z_pos = tf.math.l2_normalize(z_pos, axis=-1)

            # InfoNCE
            logits = self.model.critic_score(z_anchor, z_pos)
            logits /= temp
            labels = tf.range(B)  # label i => row i
            loss_vec = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=logits
            )
            if self.config.loss_reduction == "mean":
                info_loss = tf.reduce_mean(loss_vec)
            else:
                info_loss = tf.reduce_sum(loss_vec)
            info_loss = tf.reduce_mean(loss_vec)

            # KL
            kl_anchor = tfd.kl_divergence(z_dist_anchor, self.prior)
            kl_pos = tfd.kl_divergence(z_dist_pos, self.prior)
            kl_val = 0.5 * (tf.reduce_mean(kl_anchor) + tf.reduce_mean(kl_pos))

            total_loss = info_loss + self.beta * kl_val

        grads = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        # Update EMA prior if needed
        if self.config.prior != "isotropic_gaussian":
            both_samples = tf.concat([z_anchor, z_pos], axis=0)
            batch_mu = tf.reduce_mean(both_samples, axis=0)
            batch_var = tf.math.reduce_variance(both_samples, axis=0)
            self.ema_mu.assign(self.ema_decay * self.ema_mu + (1.0 - self.ema_decay) * batch_mu)
            self.ema_var.assign(self.ema_decay * self.ema_var + (1.0 - self.ema_decay) * batch_var)

        return total_loss, info_loss, kl_val

    @tf.function
    def _train_step_explicit(self, x_anchor, x_pos, x_neg):
        temp = 0.2
        with tf.GradientTape() as tape:
            z_dist_anchor = self.model.encode(x_anchor, use_target=False)
            z_anchor = z_dist_anchor.sample()  # [B, z_dim]
            z_dist_pos = self.model.encode(x_pos, use_target=False)
            z_pos = z_dist_pos.sample()        # [B, z_dim]

            B, K, Xd = tf.shape(x_neg)[0], tf.shape(x_neg)[1], tf.shape(x_neg)[2]
            x_neg_reshaped = tf.reshape(x_neg, [B * K, Xd])  # => [B*K, X_dim]
            z_dist_neg = self.model.encode(x_neg_reshaped, use_target=False)
            z_neg_all = z_dist_neg.sample()
            z_neg = tf.reshape(z_neg_all, [B, K, self.config.z_dim])

            anchorW = tf.matmul(z_anchor, self.model.W)
            pos_logits = tf.reduce_sum(anchorW * z_pos, axis=-1, keepdims=True)

            anchorW_expand = tf.expand_dims(anchorW, axis=1)
            neg_logits = tf.reduce_sum(anchorW_expand * z_neg, axis=-1)

            all_logits = tf.concat([pos_logits, neg_logits], axis=1)
            all_logits = all_logits / temp

            labels = tf.zeros(tf.shape(x_anchor)[0], dtype=tf.int32)
            loss_vec = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=all_logits)

            if self.config.loss_reduction == "mean":
                info_loss = tf.reduce_mean(loss_vec)
            else:
                info_loss = tf.reduce_sum(loss_vec)

            kl_anchor = tf.reduce_mean(tfd.kl_divergence(z_dist_anchor, self.prior))
            kl_pos = tf.reduce_mean(tfd.kl_divergence(z_dist_pos, self.prior))
            kl_val = 0.5 * (kl_anchor + kl_pos)
            total_loss = info_loss + self.beta * kl_val
        
        grads = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
                                       
        if self.config.prior != "isotropic_gaussian":
            both_samples = tf.concat([z_anchor, z_pos], axis=0)
            batch_mu = tf.reduce_mean(both_samples, axis=0)
            batch_var = tf.math.reduce_variance(both_samples, axis=0)
            self.ema_mu.assign(self.ema_decay * self.ema_mu + (1.0 - self.ema_decay) * batch_mu)
            self.ema_var.assign(self.ema_decay * self.ema_var + (1.0 - self.ema_decay) * batch_var)

        return total_loss, info_loss, kl_val

    @tf.function
    def soft_update(self, tau=0.005):
        """
        Soft update target encoder (momentum encoder style).
        """
        for var_t, var_o in zip(self.model.encoder_target.trainable_variables,
                                self.model.encoder.trainable_variables):
            var_t.assign(tau * var_o + (1.0 - tau) * var_t)

    def save_weights(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        self.model.save_weights(os.path.join(path, 'ib_model.weights.h5'))
        print(f"[ContextIBModel] Model weights saved to {path}")

    def load_weights(self, path):
        self.model.load_weights(path)
        print(f"[ContextIBModel] Model weights loaded from {path}")