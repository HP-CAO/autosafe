import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import dataclasses
from dataclasses import field
import numpy as np
import os


"""
=======================================================================================================
======= DEPRECATED: The following code is deprecated. Use IBmodel_core.py =============================
=======================================================================================================
"""

@dataclasses.dataclass
class IBConfig:
    epochs: int = 100
    batch_size: int = 128
    replay_buffer_size: int = 10000
    learning_rate: float = 0.001
    model_name: str = "PhyIB_model"
    context_dim: int = 3 # The dimension of the context vector
    embedding_dim: int = 10 # Use 2 if you want to visualize the latent space
    context_reparameterize: bool = False # Whether to use a separate prediction for each element of context.
    context_sensitivity: float = 0.01
    use_layer_norm: bool = True # Whether to use layer normalization in the encoder
    beta: float = 1e-4 # trade-off parameter in the information bottleneck. start with a small value
    accepctance_threshold: float = 0.1 # The threshold for the acceptance of new beta
    infoNCE_mode: str = "unpaired" # The mode of InfoNCE loss: 'unpaired' | 'paired'. TODO: implement paired mode
    prior: str = "Isotropic_Gaussian" # The prior distribution of the latent variable. Using isotropic Gaussian by default.
    loss_function: str = "IB_loss" 
    loss_reduction: str = "mean" # Specifies the reduction to apply to the output: 'mean' | 'sum'.
    num_samples: int = 20 # num of samples used for sampling
    num_negatives: int = 10 # num of negative samples uesd for infoNCE loss. (start with 1/5 of batch_size)
    model_path: str = "" 
    pretrain: bool = True # Whether to pretrain the model
    training_period: int = 1000 # The period of training the IB model

def generate_tril_dist(params, dim=2):
  """Given flat parameters, assemble a MultivariateTril Distribution."""
  loc = params[..., :dim]
  tril_params = tfp.bijectors.FillScaleTriL(
      diag_bijector=tfp.bijectors.Exp(),
      diag_shift=None).forward(params[..., dim:])
  return tfd.MultivariateNormalTriL(loc=loc, scale_tril=tril_params)

class MLPEncoder(tf.keras.layers.Layer):
    """ MLP Encoder for dimensionality reduction."""
    def __init__(self, output_dim=2, use_layer_norm=True):
        super(MLPEncoder, self).__init__()
        self.use_layer_norm = use_layer_norm
        self.output_dim = output_dim
        self.dense1 = tf.keras.layers.Dense(254, kernel_initializer='he_normal') # He initialization
        self.dense2 = tf.keras.layers.Dense(254, kernel_initializer='he_normal')
        self.dense3 = tf.keras.layers.Dense(128, kernel_initializer='he_normal')
        self.dense4 = tf.keras.layers.Dense(128, kernel_initializer='he_normal')
        self.dense5 = tf.keras.layers.Dense(128, kernel_initializer='he_normal')
        self.dense6 = tf.keras.layers.Dense(output_dim)
        self.layer_norm1 = tf.keras.layers.LayerNormalization()
        self.layer_norm2 = tf.keras.layers.LayerNormalization()
        self.layer_norm3 = tf.keras.layers.LayerNormalization()
        self.layer_norm4 = tf.keras.layers.LayerNormalization()
        self.layer_norm5 = tf.keras.layers.LayerNormalization()

        self.new_added_layers = []
            
    def call(self, x):
        # use layer normalization and residual connection
        if self.use_layer_norm:
            x = self.dense1(x)
            x = self.layer_norm1(tf.nn.gelu(x))
            x = self.dense2(x)
            x = self.layer_norm2(tf.nn.gelu(x))
            x = self.dense3(x)
            x = self.layer_norm3(tf.nn.gelu(x))
            x = self.dense4(x)
            x = self.layer_norm4(tf.nn.gelu(x))
            y = self.dense5(x)
            y = self.layer_norm5(tf.nn.gelu(y))
            x = x + tf.nn.gelu(y)
            for layer in self.new_added_layers:
                x = layer(x)
            z = self.dense6(x)
        else:
            x = self.dense1(x)
            x = tf.nn.gelu(x)
            x = self.dense2(x)
            x = tf.nn.gelu(x)
            x = self.dense3(x)  
            x = tf.nn.gelu(x)
            x = self.dense4(x)
            x = tf.nn.gelu(x)
            x = self.dense5(x)
            y = tf.nn.gelu(x)
            for layer in self.new_added_layers:
                x = layer(x)
            z = self.dense6(y)

        return z
    
    def add_new_layer(self, units):
        new_dense_layer = tf.keras.layers.Dense(units, kernel_initializer='he_normal')
        if self.use_layer_norm:
            new_layer_norm = tf.keras.layers.LayerNormalization()
            self.new_added_layers.append(lambda x: new_layer_norm(tf.nn.gelu(new_dense_layer(x))))
        else:
            self.new_added_layers.append(lambda x: tf.nn.gelu(new_dense_layer(x)))

class Context_predictor(tf.keras.layers.Layer):
    """ Context predictor.
        Args:
            context_dim: The dimension of the context vector.
            reparemeterize: whether to use a separate prediction for each element of the context vector.
    """
    def __init__(self, context_dim, embedding_dim, reparemeterize=False):
        super(Context_predictor, self).__init__()
        self.reparemeterize = reparemeterize
        self.context_dim = context_dim
        self.embedding_dim = embedding_dim

        # If the context dimension is 1, we don't need to reparameterize
        if context_dim == 1:
            self.reparemeterize = False
        
        if self.reparemeterize:
            if embedding_dim % context_dim != 0:
                raise ValueError("The embedding dimension must be a multiple of the context dimension.")
            self.dense_layers = [tf.keras.layers.Dense(16, activation='relu') for _ in range(context_dim)]
            self.final_layers = [tf.keras.layers.Dense(1, activation='linear') for _ in range(context_dim)]
        
        else:
            self.dense_layer = tf.keras.layers.Dense(16, activation='relu')
            self.final_layer = tf.keras.layers.Dense(context_dim)
    
    def call(self, embedding):
        if self.reparemeterize:
            # Split the embedding into context_dim parts according to the multiplier. e.g., context_dim=2, embedding_dim=4
            # Then we have two parts of the embedding for each context element.
            embedding_split = tf.split(embedding, self.context_dim, axis=-1)
            x_s = [dense_layer(embedding_split) for dense_layer, embedding_split in zip(self.dense_layers, embedding_split)]
            x_s = [final_layer(embedding) for final_layer, embedding in zip(self.final_layers, x_s)]
            x = tf.concat(x_s, axis=-1)
        else:
            x = self.dense_layer(embedding)
            x = self.final_layer(x)
        return x
    
class ContextID_Block(tf.keras.layers.Layer):
    """ Context ID block."""
    def __init__(self, embedding_dim=2, use_layer_norm=True):
        super(ContextID_Block, self).__init__()
        self.embedding_dim = embedding_dim 
        self.num_params = embedding_dim + (embedding_dim * (embedding_dim + 1))//2
        self.encoder = MLPEncoder(output_dim=self.num_params, use_layer_norm=use_layer_norm)

    def encode(self, x):
        params = self.encoder(x)
        return params
    
    def call(self, x):
        # return a multivariate normal distribution
        params = self.encode(x)
        return generate_tril_dist(params, dim=self.embedding_dim)

class PhyIB_Block(tf.keras.Model):
    """ PhyIB block."""
    def __init__(self, context_dim, embedding_dim=2, context_reparameterize=False, use_layer_norm=True):
        super(PhyIB_Block, self).__init__()
        self.contextIB_layer = ContextID_Block(embedding_dim=embedding_dim, use_layer_norm=use_layer_norm)
        self.context_predictor = Context_predictor(context_dim=context_dim, embedding_dim=embedding_dim, reparemeterize=context_reparameterize)

    def encode(self, x):
        z_dist = self.contextIB_layer(x)
        return z_dist
    
    def call(self, x):
        # x shape options: (batch_size, input_dim), (input_dim,), or (batch_size, num_context, input_dim)
        reshape = False
        batch_size = x.shape[0]
        if len(x.shape) == 1:
            x = tf.expand_dims(x, axis=0)
        if len(x.shape) > 2:
            # flatten into (batch_size * num_context, input_dim) first
            x = tf.reshape(x, (-1, x.shape[-1]))
            reshape = True
        z_dist = self.encode(x)
        z = z_dist.sample()
        context_pred = self.context_predictor(z)

        if reshape:
            # convert back to (batch_size, num_context, context_dim)
            context_pred = tf.reshape(context_pred, (batch_size, -1, context_pred.shape[-1]))

        return context_pred

class PhyIBModel:
    def __init__(self, param: IBConfig):
        self.params = param
        self.model = PhyIB_Block(context_dim=param.context_dim,
                                embedding_dim=param.embedding_dim, 
                                context_reparameterize=param.context_reparameterize,
                                use_layer_norm=param.use_layer_norm)

        self.marginal = tfd.MultivariateNormalDiag(loc=tf.zeros(param.embedding_dim))
        self.beta = param.beta
        self._optimizer = tf.keras.optimizers.Adam(learning_rate=param.learning_rate)

        if param.model_path != "":
            print(f"Loading model from {param.model_path}")
            self.model.load_weights(param.model_path)
        else:
            print("IB model initialized.")
    
    def add_layer_to_encoder(self, units):
        self.model.contextIB_layer.encoder.add_new_layer(units)
        
    @tf.function
    def _optimize(self, x, context, x_neg=None):
        with tf.GradientTape() as tape:
            context_pred = self.model(x)
            context_contrasive = self.model(x_neg) if x_neg is not None else None
            loss_info = self.infoNCE_loss(context, context_pred, 
                                          context_contrasive, 
                                          negative_mode=self.params.infoNCE_mode, 
                                          reduction=self.params.loss_reduction)

            z_dist = self.model.encode(x)
            rate = self.rate(z_dist)
            loss = loss_info + self.beta * rate

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self._optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return {'total_loss': loss, 'infoNCE_loss': loss_info, 'rate': rate}

    def optimize(self, train_data_batch_pos, train_data_batch_neg=None):
        x_pos, context_pos = train_data_batch_pos[0], train_data_batch_pos[1]
        x_pos = tf.convert_to_tensor(x_pos, dtype=tf.float32)
        context_pos = tf.convert_to_tensor(context_pos, dtype=tf.float32)

        if train_data_batch_neg is not None:
            x_neg, context_neg = train_data_batch_neg[0], train_data_batch_neg[1] 
            x_neg = tf.convert_to_tensor(x_neg, dtype=tf.float32)
            context_neg = tf.convert_to_tensor(context_neg, dtype=tf.float32)
        else:
            x_neg = None

        loss = self._optimize(x_pos, context_pos, x_neg)
        return loss
    
    def rate(self, z_dist):
        """ Compute the rate loss (KL divergence between the posterior and the prior)."""
        reduction = self.params.loss_reduction
        num_samps = self.params.num_samples
        z_samples = z_dist.sample((num_samps,) if num_samps > 1 else ())
        tmp = tf.reduce_mean(z_dist.log_prob(z_samples) - self.marginal.log_prob(z_samples), axis=0)
        if reduction == 'mean':
            return tf.reduce_mean(tmp)
        elif reduction == 'sum':
            return tf.reduce_sum(tmp)
        else:
            raise ValueError(f"Invalid reduction: {reduction}") 
        
    def infoNCE_loss(self, context, context_pred, context_contrasive, reduction='sum'):
        # Compute InfoNCE loss for a query and positive and negative keys.

        positive_score = tf.reduce_sum(context * context_pred, axis=-1, keepdims=True)
        context_ = tf.expand_dims(context, 1)
        negative_score = tf.matmul(context_, context_contrasive, transpose_b=True)
        negative_score = tf.squeeze(negative_score, axis=1)

        logits = tf.concat([positive_score, negative_score], axis=1)
        labels = tf.zeros(shape=(logits.shape[0],), dtype=tf.int64)

        temperature = 0.5
        class_err = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits/temperature)

        if reduction == 'mean':
            return tf.reduce_mean(class_err)
        elif reduction == 'sum':
            return tf.reduce_sum(class_err)
        # return infoNCE_loss(context, context_pred, context_contrasive, negative_mode, normalize, reduction)
    
    @tf.function
    def sample_z(self, x, mode='sample'):
        # Sample z from the posterior distribution. x shape: (batch_size, input_dim)
        z_dist = self.model.encode(x)

        if mode == 'sample':
            # sample once
            z_sample = z_dist.sample()
            return z_sample
        elif mode == 'sample_low_var':
            # sample multiple times and take the mean to reduce the variance
            num_samples = self.params.num_samples
            z_samples = z_dist.sample((num_samples,))
            z_sample = tf.reduce_mean(z_samples, axis=0)
            return z_sample
        elif mode == 'mean':
            # compute the mean of the samples
            num_samples = self.params.num_samples
            z_samples = z_dist.sample((num_samples,))
            z_sample = tf.reduce_mean(z_samples, axis=0)
            z_mean = tf.reduce_mean(z_sample, axis=0)
            return tf.squeeze(z_mean) # kill the dimension of 1
        else:
            raise ValueError(f"Invalid mode: {mode}")
    
    def sample_z_prior(self, num_samples=1):
        z_samples = self.marginal.sample((num_samples,))
        return tf.squeeze(z_samples)
    
    def predict_context(self, x, mode='sample'):
        """ Predict the context. 
            Use predicted context if downstream task requires more information."""
        z_samples = self.sample_z(x, mode=mode)
        context_pred = self.model.context_predictor(z_samples)
        return context_pred
    
    """ Use large batch size to compute the MI estimate.
        The MI estimate is upper bounded by log K, so the bound will be loose if MI > log K.
        see https://arxiv.org/abs/1807.03748 """
    def compute_MI_estimate(self, minibatch, num_samples=30):
        X, Y = minibatch[0], minibatch[1]
        K = X.shape[0]
        X = tf.convert_to_tensor(X, dtype=tf.float32)
        Y = tf.convert_to_tensor(Y, dtype=tf.float32)
        c_pred = self.model(X, mode='sample')
        infoNCE_loss = self.infoNCE_loss(Y, c_pred, reduction='mean')
        MI_estimate = np.log(K) - infoNCE_loss.numpy()
        return MI_estimate
    
    def update_beta(self):
        """TODO: Implement the beta update rule."""
        pass

    def save_weights(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        self.model.save_weights(os.path.join(path, "context.weights.h5"))
    
    def get_closest_context(self, query_batch, context_list, num_closest=1):
        """ Get the closest context to the predicted context
            Args:
                query_batch: A tuple of (X, c) where X is the input and c is the ground truth context.
                context_list: A list of context vectors.
                num_closest: The number of closest contexts to return.
        """
        context_list = tf.convert_to_tensor(context_list, dtype=tf.float32)

        X, c = query_batch[0], query_batch[1]
        X = tf.convert_to_tensor(X, dtype=tf.float32)
        c_query = self.model(X)

        c_query_expanded = tf.expand_dims(c_query, 1)
        dot_prod = tf.matmul(c_query_expanded, context_list, transpose_b=True)
        dot_prod = tf.squeeze(dot_prod, axis=1)

        # reorder the context_list based on the similarity score
        idx = tf.argsort(dot_prod, direction='DESCENDING')
        context_list = tf.gather(context_list, idx)
        context_list = tf.squeeze(context_list)

        return context_list[:num_closest].numpy()
    

def infoNCE_loss(query, positive_keys, 
                 negative_keys=None, negative_mode='unpaired', normalize=False,
                 reduction='mean'):
    """Compute InfoNCE loss for a query and positive and negative keys.

    References: 
        https://arxiv.org/pdf/1807.03748
        Modified based on https://github.com/RElbers/info-nce-pytorch

    Args:
        query: A tensor of shape (batch_size, embedding_dim) representing the query.
        positive_keys: A tensor of shape (batch_size, embedding_dim) representing the positive keys.
        negative_keys (optional): A tensor of negative keys
            If negative_mode == 'unpaired': A tensor of shape (num_negatives, embedding_dim).
            If negative_mode == 'paired': A tensor of shape (batch_size, num_negatives, embedding_dim).
            If None, then the negative keys for a sample are the positive keys for the other samples.
        normalize: Whether to normalize the embeddings before computing the dot product.
            If True, the cosine similarity is used as scores.
            If False, log-bilinear model is used as scores.
        reduction: Specifies the reduction to apply to the output: 'mean' | 'sum'.
    """
     # Check input dimensionality.
    if len(query.shape) != 2:
        raise ValueError('<query> must have 2 dimensions.')
    if len(positive_keys.shape) != 2:
        raise ValueError('<positive_key> must have 2 dimensions.')
    if negative_keys is not None:
        if negative_mode == 'unpaired' and len(negative_keys.shape) != 2:
            raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
        if negative_mode == 'paired' and len(negative_keys.shape) != 3:
            raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

    # Check matching number of samples.
    if len(query) != len(positive_keys):
        raise ValueError('<query> and <positive_key> must must have the same number of samples.')
    if negative_keys is not None:
        if negative_mode == 'paired' and len(query) != len(negative_keys):
            raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

    # check tf types of query and keys
    if query.dtype != tf.float32:
        query = tf.cast(query, tf.float32)

    # If using normalized embeddings, the cosine similarity if used as scores.
    if normalize:
        query = tf.keras.layers.UnitNormalization(axis=-1)(query)
        positive_keys = tf.keras.layers.UnitNormalization(axis=-1)(positive_keys)
        if negative_keys is not None:
            negative_keys = tf.keras.layers.UnitNormalization(axis=-1)(negative_keys)
    
    if negative_keys is not None:
        # Compute positive scores of shape (batch_size, 1).
        positive_dot_prod = tf.reduce_sum(query * positive_keys, axis=1, keepdims=True)

        # Compute negative scores of shape (batch_size, num_negatives).
        if negative_mode == 'unpaired':
            negative_dot_prod = tf.matmul(query, tf.transpose(negative_keys))
        elif negative_mode == 'paired':
            query_expanded = tf.expand_dims(query, 1)
            negative_dot_prod = tf.matmul(query_expanded, negative_keys, transpose_b=True)
            negative_dot_prod = tf.squeeze(negative_dot_prod, axis=1)
        
        logits = tf.concat([positive_dot_prod, negative_dot_prod], axis=1) # Shape (batch_size, num_negatives + 1)

        # First index in last dimension are the positive samples.
        labels = tf.zeros(shape=(logits.shape[0],), dtype=tf.int64)

    else:
        # Compute the dot product of the query with the positive keys.
        logits = tf.matmul(query, positive_keys, transpose_b=True)

        # The labels are the indices of the positive keys.
        labels = tf.range(logits.shape[0], dtype=tf.int64)
    
    """ Temperature: The temperature parameter for the softmax function. Used to smooth the distribution.
        Smaller values of temperature result in a distribution that is closer to one-hot; 
        Larger values of temperature result in a softer distribution. """    

    temperature = 1.0
    class_err = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits/temperature)

    if reduction == 'mean':
        return tf.reduce_mean(class_err)
    elif reduction == 'sum':
        return tf.reduce_sum(class_err)
    else:
        raise ValueError(f"Invalid reduction: {reduction}")
    