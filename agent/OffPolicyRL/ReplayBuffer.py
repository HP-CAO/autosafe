import pickle
import numpy as np
import copy
import random
from collections import deque, defaultdict, namedtuple
import tensorflow as tf


def shape(exp):
    if type(exp) is list:
        return len(exp)
    if type(exp) is np.ndarray:
        try:
            return len(exp)
        except:
            return 1  # for 1D np.ndarray
    else:
        return 1


def type_of(exp):
    if type(exp) is bool:
        return bool
    else:
        return float


class ReplayMemory:
    """
    Replay memory class to store trajectories
    """

    def __init__(self, size, combined_experience_replay=False):
        """
        initializing the replay memory
        """
        self.combined_experience_replay = combined_experience_replay
        self.new_head = False
        self.k = 1
        self.head = -1
        self.full = False
        self.size = int(size)
        self.memory = None
        self.initialized = False

    def initialize(self, experience):

        self.memory = [np.zeros(shape=(self.size, shape(exp)), dtype=type_of(exp)) for exp in experience]
        self.memory.append(np.zeros(shape=(self.size, 1), dtype=float))  # this is for the priority sampling

    def add(self, experience):
        if self.memory is None:
            self.initialize(experience)
            self.initialized = True

        if len(experience) + 1 != len(self.memory):
            raise Exception('Experiment not the same size as memory', len(experience), '!=', len(self.memory))

        for e, mem in zip(experience, self.memory):
            mem[self.k] = e

        self.head = self.k
        self.new_head = True
        self.k += 1
        if self.k >= self.size:
            self.k = 1  # replace the oldest one with the latest one
            self.full = True

    def sample(self, batch_size):

        if not self.full:
            if self.k < batch_size:
                batch_size = self.k
                # raise Exception(f'Not enough samples to sample: batch_size ({batch_size}) > current size {self.k}')
            r = self.k
        else:
            r = self.size

        random_idx = np.random.choice(r, size=batch_size, replace=False)

        if self.combined_experience_replay:
            if self.new_head:
                random_idx[0] = self.head  # always add the latest one
                self.new_head = False

        return [mem[random_idx] for mem in self.memory]

    def sample_recent(self, batch_size):
        if not self.full:
            if self.k < batch_size:
                raise Exception(f'Not enough samples to sample: batch_size ({batch_size}) > current size {self.k}')
            idx = np.arange(self.k - batch_size, self.k)
        else:
            if self.k < batch_size:
                idx_1 = np.arange(0, self.k)
                idx_2 = np.arange(self.size - (batch_size - self.k), self.size)
                idx = np.hstack((idx_1, idx_2))
            else:
                idx = np.arange(self.k - batch_size, self.k)
        return [mem[idx] for mem in self.memory]

    def get(self, start, length):
        return [mem[start:start + length] for mem in self.memory]

    def get_size(self):
        if self.full:
            return self.size
        return self.k

    def get_max_size(self):
        return self.size

    def reset(self):
        self.k = 0
        self.head = -1
        self.full = False
        self.memory = None
        self.new_head = False

    def shuffle(self):
        """
        to shuffle the whole memory
        """
        self.memory = self.sample(self.get_size())

    def save2file(self, file_path):
        with open(file_path, 'wb') as fp:
            pickle.dump(self.memory, fp)

    def load_memory_caches(self, path):

        with open(path, 'rb') as fp:
            memory = pickle.load(fp)
            if self.memory is None:
                self.memory = memory
            else:
                self.memory = np.hstack((self.memory, memory))

        print("Load memory caches, pre-filled replay memory!")


""" Contextual Replay Memory """


class ContextualReplayBuffer:
    def __init__(self, buffer_size, context_horizon=10, name='ContextualReplayMemory'):
        """
            Args:
                buffer_size: int, size of the replay buffer of each context
                context_horizon: int, number of transitions to form an input for context inference
                name: str, name of the ContextualReplayMemory
        """
        self.context_dict = defaultdict(list)
        self.memory = defaultdict(list)
        self.buffer_size = buffer_size
        self.name = name
        self.num_transitions = context_horizon
        self.num_contexts = 0

    def _context_exists(self, c_gt):
        # check if the context exists in the context_dict
        for idx, context in self.context_dict.items():
            if np.allclose(context, c_gt, atol=1e-3):
                return idx
        return None

    def sample_context(self, num_contexts=2):
        # sample context from the context_dict
        if len(self.context_dict) < num_contexts:
            raise ValueError(f"Number of contexts in context_dict is less than {num_contexts}")
        idx = np.random.choice(list(self.context_dict.keys()), num_contexts, replace=False)
        return [self.context_dict[i] for i in idx]

    def add(self, c_gt, experience):
        idx = self._context_exists(c_gt)
        if idx is not None:
            self.memory[idx].add(experience)
        else:
            print(
                f'Adding new context {c_gt} to {self.name}')  # if the context is not in the dictionary, add the context and create a new ReplayMemory
            idx = len(self.context_dict)
            self.context_dict[idx] = c_gt
            self.memory[idx] = ReplayMemory(self.buffer_size)
            self.memory[idx].add(experience)
            self.num_contexts += 1

    def sample_rl_batch(self, batch_size, c_gt=None):
        # sample a batch of transitions from a context to train the RL agent
        if c_gt is None:
            # randomly sample a context
            idx = np.random.choice(list(self.context_dict.keys()))
        else:
            idx = self._context_exists(c_gt)
            if idx is None:
                raise ValueError(f"Context {c_gt} not found in context_dict")
        mem = self.memory[idx]
        if mem.k < batch_size:
            batch_size = mem.k
            if batch_size == 0:
                batch_size = 1
        return mem.sample(batch_size)

    def sample_contrastive_batch(self, batch_size, mode='in_batch'):
        """
            Data Batch = (stacked transitions from a context, context label)
            Args:
                batch_size: int, size of the batch
                mode: str, 'in_batch' or 'explicit_negatives'. If 'in_batch', mix the positive and negative samples in the same batch.
                                If 'explicit_negatives', sample negative samples explicitly.
        """
        context_2use = self.sample_context(num_contexts=self.num_contexts)
        if mode == 'in_batch':
            batch_size_context = batch_size // self.num_contexts
            X = []
            lbs = []
            for c in context_2use:
                batch = self.sample_transitions(batch_size_context, c_gt=c)
                X.append(batch[0])
                lbs.append(batch[1])
            X = np.concatenate(X, axis=0)
            lbs = np.concatenate(lbs, axis=0)
            return (X, lbs)

        elif mode == 'explicit_negatives':
            c_pos = context_2use[0]
            c_neg = context_2use[1:]
            num_neg = len(c_neg)
            X_pos, lbs_pos = self.sample_transitions(batch_size, c_gt=c_pos)
            X_neg = []
            lbs_neg = []
            for c in c_neg:
                batch = self.sample_transitions(batch_size, c_gt=c)
                X_neg.append(batch[0])
                lbs_neg.append(batch[1])
            X_neg = np.reshape(X_neg, (batch_size, num_neg, -1))
            lbs_neg = np.reshape(lbs_neg, (batch_size, num_neg, -1))
            return (X_pos, lbs_pos), (X_neg, lbs_neg)

        # if c_pos is None:
        #     c_pos = self.sample_context(1)[0]

        # pos_batch = self.sample_transitions(batch_size, c_gt=c_pos)
        # c_list = list(self.context_dict.values())
        # if len(c_list) - 1 == 0:
        #     raise ValueError("Only one context found in context_dict")

        # X = []
        # lbs = []
        # ct = 0
        # random.shuffle(c_list) # shuffle the context list
        # for c in c_list:
        #     if np.allclose(c, c_pos, atol=1e-3):
        #         continue
        #     neg_batch = self.sample_transitions(batch_size, c_gt=c)
        #     X.append(neg_batch[0])
        #     lbs.append(neg_batch[1])
        #     ct += 1
        #     if ct == num_neg:
        #         break

        # num_neg = len(X)

        # # reshape to (batch_size, num_neg, feature_dim)
        # X = np.reshape(X, (batch_size, num_neg, -1))
        # lbs = np.reshape(lbs, (batch_size, num_neg, -1))
        # neg_batch = (X, lbs)
        # return pos_batch, neg_batch

    def sample_transitions(self, batch_size, c_gt=None):
        # sample stacked transitions from a context
        num_transitions = self.num_transitions
        if c_gt is None:
            idx = np.random.choice(list(self.context_dict.keys()))
        else:
            idx = self._context_exists(c_gt)
            if idx is None:
                raise ValueError(f"Context {c_gt} not found in context_dict")
        mem = self.memory[idx]

        data_batch = mem.sample(int(batch_size * num_transitions))
        X, Y = data_batch[0], data_batch[1]
        Y_elem = Y[0, :]
        X_dim = X.shape[1]
        X = X.reshape((batch_size, num_transitions * X_dim))  # stack the transitions
        Y = np.tile(Y_elem, (batch_size, 1))  # repeat the context label
        return (X, Y)

    def check_context(self, c1, c2):
        if np.linalg.norm(c1 - c2, ord=1) < self.sensitivity:
            return True

    def create_copy(self, name):
        # Create a new instance of the ContextualReplayMemory with the same parameters
        new_copy = ContextualReplayBuffer(
            buffer_size=self.buffer_size,
            sensitivity=self.sensitivity,
            name=name
        )

        # Deep copy the context_dict to ensure it is independent
        new_copy.context_dict = copy.deepcopy(self.context_dict)

        # Deep copy the memory, ensuring each ReplayMemory instance is also deeply copied
        new_copy.memory = defaultdict(list, {
            idx: copy.deepcopy(replay_memory) for idx, replay_memory in self.memory.items()
        })

        return new_copy

    def save2file(self, file_path):
        with open(file_path, 'wb') as fp:
            pickle.dump(self.context_dict, fp)
            pickle.dump(self.memory, fp)

    def load_memory_caches(self, path):
        with open(path, 'rb') as fp:
            self.context_dict = pickle.load(fp)
            self.memory = pickle.load(fp)


class ContextSampler:

    def __init__(self, context_dict, horizon=1e5, temperature=1.0, logger=True):
        """
        Initialize the ContextSampler with a context dictionary, a reward horizon, and a temperature.

        Parameters:
        - context_dict: A dictionary where keys are indices and values are the corresponding contexts.
        - horizon: The maximum number of rewards to keep track of for each context.
        - temperature: Controls the sharpness of the probability distribution for sampling: higher temperature -> more uniform distribution.
        """
        self.horizon = horizon
        self.context_dict = context_dict
        self.context_list = list(context_dict.values())
        self.context_indices = list(context_dict.keys())

        # Initialize the rewards and priority dictionaries
        self.rewards = {c: deque(maxlen=int(horizon)) for c in
                        self.context_indices}  # keep track of rewards for each context
        self.priority = {c: 1 / len(self.context_indices) for c in
                         self.context_indices}  # start with UNIFORM distribution
        self.temperature = temperature  # higher temperature -> more uniform distribution

        self.min_prob = 0.1
        self.max_prob = 0.9

        if logger:
            self.logger = True
            self.C_samples = deque(maxlen=500)

    def add(self, c_gt, reward):
        idx = self._context_exists(c_gt)
        if idx is not None:
            self.rewards[idx].append(reward)
            self.update_sampling_probabilities()
        else:
            raise ValueError(f"Context {c_gt} not found in context_dict")

    def update_sampling_probabilities(self):
        # Calculate the average reward for each context
        avg_rewards = {idx: np.mean(self.rewards[idx]) if len(self.rewards[idx]) > 0 else 1e-2 for idx in
                       self.context_indices}

        # NOTE: we want to prioritize contexts with LOW average rewards
        scaled_rewards = np.array([-avg_rewards[idx] for idx in self.context_indices]) / self.temperature

        # Apply the softmax function
        exp_rewards = np.exp(scaled_rewards - np.max(scaled_rewards))  # subtract max for numerical stability
        sum_exp_rewards = np.sum(exp_rewards)

        probs = exp_rewards / sum_exp_rewards
        min_prob = self.min_prob
        max_prob = self.max_prob
        probs = np.clip(probs, min_prob, max_prob)
        probs = probs / np.sum(probs)

        # Update the priority (sampling probability) for each context
        self.priority = {idx: probs[i] for i, idx in enumerate(self.context_indices)}

    def sample(self):
        ind = np.random.choice(self.context_indices, p=list(self.priority.values()))
        c_sample = self.context_dict[ind]
        if self.logger:
            self.C_samples.append(c_sample[1])
        return c_sample

    def _context_exists(self, c_gt):
        for idx, context in self.context_dict.items():
            if np.array_equal(context, c_gt):
                return idx
        return None

    def _samples_hist(self):
        hist = np.histogram(self.C_samples, bins=10)
        return hist

    def reset(self):
        self.rewards = {c: deque(maxlen=int(self.horizon)) for c in self.context_indices}
        self.priority = {c: 1 / len(self.context_indices) for c in self.context_indices}