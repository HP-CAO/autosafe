import logging
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
from dataclasses import dataclass, field


"""
=======================================================================================================
======= DEPRECATED: The following code is deprecated. Use IBmodel_core.py =============================
=======================================================================================================
"""

@dataclass
class IBConfig:
    batch_size: int = 512
    learning_rate: float = 0.0005
    model_name: str = "PhyIB_model"
    context_dim: int = 1  # Dimension of the context vector
    embedding_dim: int = 2  # Use 2 if you want to visualize the latent space
    use_layer_norm: bool = True  # Whether to use layer normalization in the encoder
    beta: float = 0.001  # Initial trade-off parameter in the information bottleneck
    prior: str = "other"  # Prior distribution for the latent variable: 'isotropic_gaussian' | 'other' for aggregated posterior
    loss_reduction: str = "mean"  # Reduction to apply to the output: 'mean' | 'sum'
    model_path: str = ""
    training_period: int = 5000  # Period of training the IB model
    infoNCE_mode: str = "in_batch"  # 'in_batch' or 'explicit_negatives'
    num_negatives: int = 5  # Number of negative samples used for explicit negative pairing
    ema_decay: float = 0.99  # Decay rate for EMAs of posterior mean and variance  
    embedding_type: str = "MLP"  # Type of embedding layer: 'MLP' | 'Linear' for bilinear critic
    encoder_hidden_units: list = field(default_factory=lambda: [128, 256, 256, 128])  # Hidden units in the encoder

def generate_tril_dist(params, dim=2):
    """Given flat parameters, assemble a MultivariateTriL Distribution."""
    loc = params[..., :dim]
    tril_params = tfp.bijectors.FillScaleTriL(
        diag_bijector=tfp.bijectors.Exp()).forward(params[..., dim:])
    return tfd.MultivariateNormalTriL(loc=loc, scale_tril=tril_params)

class MLPEncoder(tf.keras.layers.Layer):
    """MLP Encoder for dimensionality reduction."""
    def __init__(self, output_dim=2, use_layer_norm=True, hidden_units=None):
        super(MLPEncoder, self).__init__()
        self.use_layer_norm = use_layer_norm
        self.output_dim = output_dim
        if hidden_units is not None:
            self.hidden_units = hidden_units
        else:
            self.hidden_units = [128, 128, 128]
        self.layers_list = []
        for units in self.hidden_units:
            dense_layer = tf.keras.layers.Dense(
                units, activation='gelu', kernel_initializer='he_normal')
            self.layers_list.append(dense_layer)
            if self.use_layer_norm:
                self.layers_list.append(tf.keras.layers.LayerNormalization())
        self.output_layer = tf.keras.layers.Dense(output_dim)
    
    def call(self, x):
        for layer in self.layers_list:
            x = layer(x)
        x = self.output_layer(x)
        return x

class ContextEmbeddingLayer(tf.keras.layers.Layer):
    """Context embedding layer mapping context c to embedding z."""
    def __init__(self, embedding_dim, embedding_type):
        super(ContextEmbeddingLayer, self).__init__()
        self.layers_list = []
        hiden_units = [16,16]

        if embedding_type == 'MLP':
            activation = 'gelu'
            for unit in hiden_units:
                self.layers_list.append(tf.keras.layers.Dense(unit, activation=activation))
                self.output_layer = tf.keras.layers.Dense(embedding_dim)

        if embedding_type == 'Linear':
            activation = None
            for unit in hiden_units:
                self.layers_list.append(tf.keras.layers.Dense(unit, activation=activation))
                self.output_layer = tf.keras.layers.Dense(embedding_dim)
        
    def call(self, c):
        x = c
        for layer in self.layers_list:
            x = layer(x)
        x = self.output_layer(x)
        return x

class PhyIBBlockEmbedding(tf.keras.Model):
    """PhyIB Block combining encoder and context embedding layer."""
    def __init__(self, config: IBConfig):
        super(PhyIBBlockEmbedding, self).__init__()
        self.embedding_dim = config.embedding_dim
        num_params = self.embedding_dim + (self.embedding_dim * (self.embedding_dim + 1)) // 2
        self.encoder = MLPEncoder(
            output_dim=num_params, use_layer_norm=config.use_layer_norm, hidden_units=config.encoder_hidden_units)
        self.context_embedding_layer = ContextEmbeddingLayer(
            embedding_dim=config.embedding_dim, embedding_type=config.embedding_type)
    
    def encode(self, x):
        params = self.encoder(x)
        return generate_tril_dist(params, dim=self.embedding_dim)
    
    def get_target_embedding(self, c):
        z_tar = self.context_embedding_layer(c)
        return z_tar

def info_nce_loss(queries, positives, negatives=None, temperature=0.5,
                  reduction='mean', mode='in_batch', normalize=False):
    """
    Computes the InfoNCE loss.

    Args:
        queries: Tensor of shape [batch_size, dim]
        positives: Tensor of shape [batch_size, dim]
        negatives: If mode is 'explicit_negatives', Tensor of shape [batch_size, num_negatives, dim]
        temperature: Temperature scaling for logits
        reduction: 'mean' or 'sum'
        mode: 'in_batch' or 'explicit_negatives'
        normalize: Whether to normalize embeddings before computing similarities
    Returns:
        loss: scalar tensor
    """
    # Normalize the embeddings
    if normalize:
        queries = tf.math.l2_normalize(queries, axis=-1)
        positives = tf.math.l2_normalize(positives, axis=-1)
        if negatives is not None:
            negatives = tf.math.l2_normalize(negatives, axis=-1)


    if mode == 'in_batch':
        # Use in-batch negatives
        logits = tf.matmul(queries, tf.transpose(positives)) / temperature
        # Labels are indices of the diagonal elements (positive samples)
        batch_size = tf.shape(queries)[0]
        labels = tf.range(batch_size)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits)

    elif mode == 'explicit_negatives':
        # Positives: [batch_size, dim]
        # Negatives: [batch_size, num_negatives, dim]
        # Expand queries and positives to match negatives
        queries = tf.expand_dims(queries, axis=1)  # [batch_size, 1, dim]
        positives = tf.expand_dims(positives, axis=1)  # [batch_size, 1, dim]
        # Concatenate positives and negatives
        embeddings = tf.concat([positives, negatives], axis=1)  # [batch_size, 1 + num_negatives, dim]
        # Compute similarity scores
        logits = tf.matmul(queries, embeddings, transpose_b=True)  # [batch_size, 1, 1 + num_negatives]
        logits = tf.squeeze(logits, axis=1)  # [batch_size, 1 + num_negatives]
        logits = logits / temperature
        labels = tf.zeros(tf.shape(queries)[0], dtype=tf.int32)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits)
    else:
        raise ValueError(f"Invalid mode: {mode}")
    
    if reduction == 'mean':
        return tf.reduce_mean(loss)
    elif reduction == 'sum':
        return tf.reduce_sum(loss)
    else:
        return loss

class PhyIBModel:
    def __init__(self, config: IBConfig):
        self.config = config
        self.model = PhyIBBlockEmbedding(config)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
        self.beta = config.beta
        self.infoNCE_mode = config.infoNCE_mode

        # isotropic Gaussian prior
        if config.prior == 'isotropic_gaussian':
            self.prior = tfd.MultivariateNormalDiag(
                loc=tf.zeros(config.embedding_dim),
                scale_diag=tf.ones(config.embedding_dim))
        else:
            # using aggregated posterior as prior
            self.ema_mu = tf.Variable(tf.zeros(config.embedding_dim), trainable=False)
            self.ema_var = tf.Variable(tf.ones(config.embedding_dim), trainable=False)
            self.ema_decay = config.ema_decay
            self.prior = tfd.MultivariateNormalDiag(loc=self.ema_mu, scale_diag=tf.sqrt(self.ema_var))
        
    def sample_z(self, data_batch, mode='posterior', return_type=np.array):
        data_batch = [tf.convert_to_tensor(data) for data in data_batch]
        x, c = data_batch
        z_dist = self.model.encode(x)

        if mode == 'posterior':
            z_sample = z_dist.sample()
        elif mode == 'posterior_mean':
            z_samples = z_dist.sample(20)
            z_sample = tf.reduce_mean(z_samples, axis=0)
        elif mode == 'prior':
            z_sample = self.prior.sample(tf.shape(x)[0])
        else:
            raise ValueError(f"Invalid mode: {mode}")

        return z_sample.numpy() if return_type == np.array else z_sample
    
    def optimize(self, data_batch, negative_batch=None):
        # convert data to tensors
        data_batch = [tf.convert_to_tensor(data) for data in data_batch]
        if negative_batch is not None:
            negative_batch = [tf.convert_to_tensor(data) for data in negative_batch]
        return self._optimize(data_batch, negative_batch)

    @tf.function
    def _optimize(self, data_batch, negative_batch=None):
        x, c = data_batch
        
        if negative_batch is not None:
            x_neg, c_neg = negative_batch
        else:
            x_neg = None
            c_neg = None
        
        with tf.GradientTape() as tape:
            # Encode x to get z distribution
            z_dist = self.model.encode(x)
            z_sample = z_dist.sample()
            
            # Get target embeddings from c
            z_tar = self.model.get_target_embedding(c)
            
            # Compute InfoNCE loss
            if self.infoNCE_mode == 'in_batch':
                loss_info = info_nce_loss(
                    queries=z_sample,
                    positives=z_tar,
                    temperature=0.7,
                    reduction=self.config.loss_reduction,
                    mode='in_batch',
                    normalize=True,
                )

            elif self.infoNCE_mode == 'explicit_negatives':
                # Ensure negative samples are provided
                if x_neg is None or c_neg is None:
                    raise ValueError("Negative samples are required for 'explicit_negatives' mode.")
                
                # Get negative embeddings
                batch_size = tf.shape(x)[0]
                num_negatives = tf.shape(x_neg)[1]

                # Flatten negatives
                x_neg_flat = tf.reshape(x_neg, [batch_size * num_negatives, -1])
                c_neg_flat = tf.reshape(c_neg, [batch_size * num_negatives, -1])

                # Encode negatives
                z_neg_dist = self.model.encode(x_neg_flat)
                z_neg_sample = z_neg_dist.sample()
                z_neg_sample = tf.reshape(z_neg_sample, [batch_size, num_negatives, -1])

                # Get target embeddings from c_neg
                z_neg_tar = self.model.get_target_embedding(c_neg_flat)
                z_neg_tar = tf.reshape(z_neg_tar, [batch_size, num_negatives, -1])
                
                loss_info = info_nce_loss(
                    queries=z_sample,
                    positives=z_tar,
                    negatives=z_neg_tar,
                    temperature=0.5,
                    reduction=self.config.loss_reduction,
                    mode='explicit_negatives',
                    normalize=False
                )

            else:
                raise ValueError(f"Invalid infoNCE_mode: {self.infoNCE_mode}")
            
            # use batch mean for KL divergence
            kl_div = tfd.kl_divergence(z_dist, self.prior)

            if self.config.loss_reduction == 'mean':
                kl_div = tf.reduce_mean(kl_div)
            else:
                kl_div = tf.reduce_sum(kl_div)
            rate_loss = self.beta * kl_div
            
            # Total loss
            total_loss = loss_info + rate_loss
        
        # Compute gradients and apply optimizer
        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        # Update EMA of posterior mean and variance if using aggregated posterior
        if self.config.prior != "isotropic_gaussian":
            batch_mu = tf.reduce_mean(z_sample, axis=0)
            batch_var = tf.math.reduce_variance(z_sample, axis=0)
            self.ema_mu.assign(self.ema_decay * self.ema_mu + (1 - self.ema_decay) * batch_mu)
            self.ema_var.assign(self.ema_decay * self.ema_var + (1 - self.ema_decay) * batch_var)
        
        return {'total_loss': total_loss, 'infoNCE_loss': loss_info, 'rate_loss': rate_loss}
    
    def save_weights(self, path):
        self.model.save_weights(path)
        logging.info(f"Model weights saved to {path}")

    def load_weights(self, path):
        self.model.load_weights(path)
        logging.info(f"Model weights loaded from {path}")
