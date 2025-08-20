import numpy as np
import tensorflow as tf
import random
import os
import logging
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

def clip_or_wrap_func(a, a_min, a_max, clip_or_wrap):
    if clip_or_wrap == 0:
        return np.clip(a, a_min, a_max)
    return (a - a_min) % (a_max - a_min) + a_min
class ActionNoise:
    def __init__(self, action_dim, bounds, clip_or_wrap):
        self.action_dim = action_dim
        self.bounds = bounds
        self.clip_or_wrap = clip_or_wrap

    def sample(self) -> np.ndarray:
        pass

    def clip_or_wrap_action(self, action):
        if len(action) == 1:
            return clip_or_wrap_func(action, self.bounds[0], self.bounds[1], self.clip_or_wrap)
        return np.array([clip_or_wrap_func(a, self.bounds[0][k], self.bounds[1][k], self.clip_or_wrap[k]) for k, a in
                         enumerate(action)])

    def add_noise(self, action):
        sample = self.sample()
        action = self.clip_or_wrap_action(action + sample)
        return action

class OrnsteinUhlenbeckActionNoise(ActionNoise):

    def __init__(self, action_dim, bounds=(-1, 1), clip_or_wrap=0, mu=0, theta=0.15, sigma=0.1, dt=0.04):
        super().__init__(action_dim, bounds, clip_or_wrap)
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.X = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.X = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.X) * self.dt
        dx = dx + self.sigma * np.random.randn(len(self.X)) * np.sqrt(self.dt)
        self.X = self.X + dx
        return self.X
    
def seed_everything(seed):
    """Set the seed for all random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def setup_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def setup_environment(cfg):
    """Configure environment settings based on GPU availability."""
    if not cfg.JobParams.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        physical_devices = tf.config.list_physical_devices('GPU')
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            # tf.config.optimizer.set_jit(True)  # Enable XLA
        except Exception as e:
            exit(f"GPU allocation failed: {e}")

def log_info(writer, global_step, info_dict, prefix, period=500):
    """Log information to Tensorboard."""
    if global_step % period == 0:
        for key, value in info_dict.items():
            writer.add_scalar(f"{prefix}/{key}", value, global_step)
    else:
        pass

class ContextSampler:
    def __init__(self, scales_dict, c_axis, 
                 grid_size=3, key_method='tuple', decimal_places=2, log=False):
        
        self.scale_min_train = scales_dict['train'][0]
        self.scale_max_train = scales_dict['train'][1]
        self.scale_min_val = scales_dict['val'][0]
        self.scale_max_val = scales_dict['val'][1]

        self.c_axis = c_axis
        self.c_axis_inv = (c_axis == 0)
        self.key_method = key_method
        
        # used for discrete sampling
        self.grid_size = grid_size
        self.decimal_places = decimal_places

        # logging
        if log:
            pass # implement later
    
    def sample(self, mode="discrete", train=True, return_key=True):
        if mode == "continuous":
            scale_vec = self._sample_continuous(train)
        elif mode == "discrete":
            scale_vec = self._sample_discrete(train)
        else:
            raise ValueError("Invalid sampling mode. Choose from {'continuous', 'discrete'}.")
        
        if return_key:
            return scale_vec, self._build_key(scale_vec)
        return scale_vec

    def _sample_continuous(self, train=True):
        if train:
            scale_min, scale_max = self.scale_min_train, self.scale_max_train
        else:
            scale_min, scale_max = self.scale_min_val, self.scale_max_val
        scale_vec = np.random.uniform(scale_min, scale_max, size=len(self.c_axis))
        scale_vec = np.round(scale_vec, decimals=self.decimal_places)
        scale_vec[self.c_axis_inv] = 1.0 # set scale=1.0 for inactive dimensions
        return scale_vec
    
    def _sample_discrete(self, train=True):
        if train:
            scale_min, scale_max = self.scale_min_train, self.scale_max_train
        else:
            scale_min, scale_max = self.scale_min_val, self.scale_max_val
        grid_vals = np.linspace(scale_min, scale_max, self.grid_size)
        scale_vec = np.random.choice(grid_vals, size=len(self.c_axis), replace=True)
        scale_vec = np.round(scale_vec, decimals=self.decimal_places)
        scale_vec[self.c_axis_inv] = 1.0
        return scale_vec
    
    def _build_key(self, scale_vec):
        """Build a hashable key for contextual replay buffer."""
        if self.key_method == 'round':
            scale_rounded = np.round(scale_vec, decimals=self.decimal_places)
            c_key = tuple(scale_rounded.tolist())
        elif self.key_method == 'tuple':
            c_key = tuple(scale_vec.tolist())
        else:
            raise ValueError("Invalid key_method. Choose from {'round', 'tuple'}.")
        return c_key
    
    def _log_sample(self, scale_vec):
        raise NotImplementedError("Logging functionality not implemented yet.")
    
    def get_all_contexts(self):
        """Generate all possible contexts for the given c_axis. 
           NOTE: only works for discrete sampling."""
        
        if self.grid_size <= 0:
            raise ValueError("Grid size must be greater than 0.")

        grid_vals = np.linspace(self.scale_min_train, self.scale_max_train, self.grid_size)
        scale_dim = len(self.c_axis)
        all_contexts = []
        for combo in np.array(np.meshgrid(*[grid_vals] * scale_dim)).T.reshape(-1, scale_dim):
            scale_vec = np.round(combo, decimals=self.decimal_places)
            scale_vec[self.c_axis_inv] = 1.0
            # discard if scale_vec is already in all_contexts
            if not any(np.allclose(scale_vec, context) for context in all_contexts):
                all_contexts.append(scale_vec.tolist())
        
        return all_contexts

def sample_c_scale(
    scale_min: float,
    scale_max: float,
    c_axis: np.ndarray,
    mode: str = 'continuous',
    grid_size: int = 3,
    key_method: str = 'tuple',
    decimal_places: int = 2):

    # Identify indices where c_axis is 0 => scale=1.0
    c_axis_inv = (c_axis == 0)

    # Sample the active dimensions
    if mode == 'continuous':
        scale_vec = np.random.uniform(scale_min, scale_max, size=len(c_axis))
    elif mode == 'discrete':
        grid_vals = np.linspace(scale_min, scale_max, grid_size)
        scale_vec = np.random.choice(grid_vals, size=len(c_axis), replace=True)
    else:
        raise ValueError("Invalid mode. Choose from {'continuous', 'discrete'}.")

    # For inactive dimensions, set scale=1.0
    scale_vec[c_axis_inv] = 1.0

    # Build the hashable key
    if key_method == 'round':
        scale_rounded = np.round(scale_vec, decimals=decimal_places)
        c_key = tuple(scale_rounded.tolist())
    elif key_method == 'tuple':
        c_key = tuple(scale_vec.tolist())
    else:
        raise ValueError("Invalid key_method. Choose from {'round', 'tuple'}.")

    return scale_vec, c_key

def tnse_vis(model, data_buffer, num_samples=100, output_path=None, step=None):
    context_list = data_buffer.storage.keys()
    context_list = sorted(context_list)

    # create color scale by sorting context_list
    color_scale = np.linspace(0, 1, len(context_list))
    colors = plt.cm.viridis(color_scale)

    Z_embedded = []
    Z_label = []
    for c in context_list:
        X, Y = data_buffer.sample_by_context(num_samples, c)
        z = model.sample_z(X, return_type=np.array)
        Z_embedded.append(z)
        Z_label.append(np.argmax(Y, axis=1))
    Z_embedded_raw = np.concatenate(Z_embedded, axis=0)
    Z_label = np.concatenate(Z_label, axis=0)

    perplexity = [5, 20, 50, 100]
    (fig, subplots) = plt.subplots(1, len(perplexity), figsize=(15, 5)) # 1 row, 4 columns

    for j, perp in enumerate(perplexity):
        tsne = TSNE(n_components=2, perplexity=perp, random_state=42)
        Z_embedded = tsne.fit_transform(Z_embedded_raw)
        ax = subplots[j]
        ax.set_title(f"Perplexity {perp}")
        for i in range(len(context_list)):
            ax.scatter(Z_embedded[Z_label == i, 0], Z_embedded[Z_label == i, 1], color=colors[i],
                       label=f'c={context_list[i]}', alpha=0.5)
        ax.legend()
        ax.axis('tight')
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())

    plt.tight_layout()
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if step is not None:
        plt.savefig(os.path.join(output_path, f"tsne_{step}.png"))
    else:
        plt.savefig(os.path.join(output_path, "tsne.png"))
    plt.close()

def plot_embeddings(model, data_buffer, embedding_dim=3, num_samples=100, output_path=None, step=None):
    """
    Scatter plot of the embeddings when embedding_dim <= 3.
    """
    context_list = data_buffer.storage.keys()
    context_list = sorted(context_list)

    # create color scale by sorting context_list
    color_scale = np.linspace(0, 1, len(context_list))
    colors = plt.cm.viridis(color_scale)

    Z_embedded = []
    Z_label = []
    for c in context_list:
        X, Y = data_buffer.sample_by_context(num_samples, c)
        z = model.sample_z(X, return_type=np.array)
        Z_embedded.append(z)
        Z_label.append(np.argmax(Y, axis=1))
    Z_embedded = np.concatenate(Z_embedded, axis=0)
    Z_label = np.concatenate(Z_label, axis=0)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if embedding_dim == 2:
        fig, ax = plt.subplots()
        for i in range(len(context_list)):
            ax.scatter(Z_embedded[Z_label == i, 0], Z_embedded[Z_label == i, 1], color=colors[i],
                       label=f'c={context_list[i]}', alpha=0.5)
        ax.legend()
        ax.axis('tight')
        if step is not None:
            plt.savefig(os.path.join(output_path, f"embeddings_{step}.png"))
        else:
            plt.savefig(os.path.join(output_path, "embeddings.png"))
        plt.close()
    elif embedding_dim == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in range(len(context_list)):
            ax.scatter(Z_embedded[Z_label == i, 0], Z_embedded[Z_label == i, 1], Z_embedded[Z_label == i, 2],
                       color=colors[i], label=f'c={context_list[i]}', alpha=0.5)
        ax.legend()
        ax.axis('tight')
        if step is not None:
            plt.savefig(os.path.join(output_path, f"embeddings_{step}.png"))
        else:
            plt.savefig(os.path.join(output_path, "embeddings.png"))
        plt.close()
    else:
        raise ValueError("Invalid embedding dimension. Choose from {2, 3}.")    
