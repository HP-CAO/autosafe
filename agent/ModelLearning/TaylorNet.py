import tensorflow as tf
import numpy as np
import dataclasses
import copy
from dataclasses import field


@dataclasses.dataclass
class TayLorNetConfig:
    epochs: int = 10
    evaluation_period: int = 2
    learning_rate: float = 0.001
    model_name: str = "taylor_model"
    hidden_dim: list = field(default_factory=lambda: [20, 6])
    aug_order: list = field(default_factory=lambda: [1, 1, 0])
    activation: list = field(default_factory=lambda: ["relu", "relu", "linear"])
    initializer_w: str = 'tn'
    initializer_b: str = 'uniform'
    model_path: str = ""
    taylor_editing: bool = False
    loss_function: str = "mean_squared_error"  # this is the default loss function in tf.keras.losses


class TaylorBlock(tf.keras.Model):
    def __init__(self, input_dim, output_dim, hidden_dim, aug_order, activation, initializer_w, initializer_b,
                 taylor_editing=False):
        super(TaylorBlock, self).__init__()
        # dim_list = [input_dim, params.dense_dims[0], params.dense_dims[1], output_dim]
        dim_list = get_dim_list(input_dim, hidden_dim, output_dim)
        weights_shape = exp_length(dim_list, aug_order).astype(np.int32)  # w = [dim_neurons, input_dim]
        num_layers = len(weights_shape)

        self.layer_list = []

        if taylor_editing:
            editing_matrix = get_knowledge_matrix()
        else:
            editing_matrix = [None] * num_layers

        for i in range(num_layers):
            if aug_order[i] != 0:
                aug_layer = TaylorAugmentLayer(augment_order=aug_order[i])
                self.layer_list.append(aug_layer)

            taylor_dense_layer = TaylorLayer(input_dim=weights_shape[i][1],
                                             units=weights_shape[i][0],
                                             name='taylor_dense',
                                             init_w=initializer_w,
                                             init_b=initializer_b,
                                             scale=0.1,
                                             trainable=True,
                                             activation=activation[i],
                                             editing_matrix=editing_matrix[i])

            self.layer_list.append(taylor_dense_layer)

    def call(self, inputs, training=None, mask=None):
        x = inputs
        for i in range(len(self.layer_list)):
            x = self.layer_list[i](x)
        return x


class TaylorModel:
    def __init__(self, params: TayLorNetConfig, input_shape, output_shape):
        self.params = params
        # self._inputs = tf.keras.layers.Input(shape=(input_shape,), name='input', dtype=tf.float16)
        # self._output = tf.keras.layers.Dense(output_shape, activation='linear')

        # The crafted taylor net can be wrapped as a tensorflow model using Functional API method.
        # Instead, we directly wrapper it as a tensorflow model using Subclass method.
        self.model = TaylorBlock(input_dim=input_shape, output_dim=output_shape,
                                 hidden_dim=self.params.hidden_dim,
                                 aug_order=self.params.aug_order,
                                 activation=self.params.activation,
                                 initializer_w=self.params.initializer_w,
                                 initializer_b=self.params.initializer_b,
                                 taylor_editing=self.params.taylor_editing)

        self._loss = self._build_loss_function()
        self._optimizer = tf.keras.optimizers.Adam(learning_rate=self.params.learning_rate)
        if self.params.model_path != '':
            self.model.load_weights(self.params.model_path)

    def _build_loss_function(self):
        if self.params.loss_function == "mean_squared_error":
            return tf.keras.losses.MeanSquaredError()
        else:
            raise NotImplementedError

    @tf.function
    def _optimize(self, inputs, labels):
        with tf.GradientTape() as tape:
            predictions = self.model(inputs)
            loss = self._loss(labels, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self._optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

    def optimize(self, train_dataset_batch):
        inputs = train_dataset_batch[0]
        labels = train_dataset_batch[1]
        loss = self._optimize(inputs, labels)
        return loss.numpy().mean()

    def predict(self, inputs):
        if len(np.shape(inputs)) < 2:
            # tensorflow requires the inputs has the dim of batch_size
            inputs = np.expand_dims(inputs, axis=0)
        return self.model(inputs).numpy().squeeze()  # squeeze to kill batch_size

    @tf.function
    def _validate(self, inputs, labels):
        predictions = self.model(inputs)
        loss = self._loss(labels, predictions)
        return loss

    def validate(self, val_dataset_batch):
        inputs = val_dataset_batch[0]
        labels = val_dataset_batch[1]
        loss = self._validate(inputs, labels)
        return loss.numpy().mean()

    def eval(self, inputs):
        # Evaluation for applications
        return

    def save_weights(self, path):
        self.model.save_weights(path)


class TaylorLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim, units, name, init_w, init_b, activation, scale=0.1, trainable=True,
                 editing_matrix=None):
        """
        input_dim: the dimension of the input vector
        units: the dimension of the output, corresponding to the number of neurons
        Usage example:
        if the input X is a batch of 1D vector of m dimensions X: [bs, m] and the units is n
        this layer will create a weights matrix W with the shape of [n, m] and bias vector with the shape of n
        the output will be Y = WX + b, leading to an output tensor with [bs, n]
        """
        super(TaylorLayer, self).__init__()
        weights_shape = [units, input_dim]
        bias_shape = [units]
        self.weights_variables = weight_variable(weights_shape, name=name, distribution=init_w, scale=scale,
                                                 trainable=trainable)
        self.biases_variables = bias_variable(bias_shape, name=name, distribution=init_b, trainable=trainable)
        self.activation = activation

        if editing_matrix is not None:
            self.phyweightsA = editing_matrix[0]
            self.phyweightsB = editing_matrix[1]
            self.phybiasesA = editing_matrix[2]
            self.phybiasesB = editing_matrix[3]
            self.nn_editing = True
        else:
            self.nn_editing = False

    def call(self, inputs, training=None, mask=None):
        inputs = tf.expand_dims(inputs, axis=-1)  # [bs, dim, 1]
        # Here to do network editing
        # self.weights_variables ---> # [n, dim] self.biased_variables -----> m
        # tf.linalg.matmul(self.weights_variables, inputs) -----> [bs, n, 1]
        # tf.squeeze ----> [bs, n]
        # logits -----> [bs, n]
        if self.nn_editing:
            pre_bias = self.biases_variables * tf.squeeze(self.phybiasesB)
            pre_weights_A = self.phyweightsA
            pre_weights_B = self.weights_variables * self.phyweightsB
            prev_layer_A = tf.squeeze(tf.linalg.matmul(pre_weights_A, inputs), axis=-1) + tf.squeeze(self.phybiasesA)
            prev_layer_B = tf.squeeze(tf.linalg.matmul(pre_weights_B, inputs), axis=-1) + pre_bias

            if self.activation == 'sigmoid':
                prev_layer_B = tf.sigmoid(prev_layer_B)
            elif self.activation == 'relu':
                prev_layer_B = tf.nn.relu(prev_layer_B)
            elif self.activation == 'tanh':
                prev_layer_B = tf.nn.tanh(prev_layer_B)
            y = prev_layer_A + prev_layer_B
        else:
            logits = tf.squeeze(tf.linalg.matmul(self.weights_variables, inputs), axis=-1) + self.biases_variables

            if self.activation == 'sigmoid':
                y = tf.sigmoid(logits)
            elif self.activation == 'relu':
                y = tf.nn.relu(logits)
            elif self.activation == 'lin':
                aa = 0.001
                bb = 100
                y = tf.multiply(logits, aa) + bb
            elif self.activation == 'tanh':
                y = tf.nn.tanh(logits)
            else:
                y = logits
        return y


class TaylorAugmentLayer(tf.keras.layers.Layer):
    def __init__(self, augment_order):
        """
        the augment order starts from 1, 1 means not augmenting
        """
        super(TaylorAugmentLayer, self).__init__()
        self.order = augment_order

    def call(self, inputs, *args, **kwargs):

        input_exp = inputs
        _, c = inputs.shape
        exp_list = [input_exp]
        index_list = np.array(range(c))
        index_next_order = np.zeros(shape=c + 1, dtype=int)

        for i in range(self.order):
            augment_list = []
            num_aug_term = 0

            for j in range(c):
                exp_tensor = exp_list[-1]
                ind_start = index_list[j]
                N, len_pre = exp_tensor.shape
                input_variable = tf.reshape(inputs[:, j], shape=(N, -1))  # n, 1
                exp_tensor_pre_slice = tf.reshape(exp_tensor[:, ind_start:], shape=(N, -1))
                exp_tem = input_variable * exp_tensor_pre_slice
                num_aug_term += exp_tensor_pre_slice.shape[-1]
                index_next_order[j + 1] = num_aug_term
                augment_list.append(exp_tem)

            index_list = copy.deepcopy(index_next_order)
            augment_tensors = tf.concat(augment_list, axis=-1)
            exp_list.append(augment_tensors)

        exp_all = tf.concat(exp_list, axis=-1)
        return exp_all


def weight_variable(shape, name, distribution, scale=0.1, trainable=True):
    if distribution == 'tn':
        initial = tf.random.truncated_normal(shape, stddev=scale, dtype=tf.float32)
    elif distribution == 'xavier':
        scale = 4 * np.sqrt(6.0 / (shape[0] + shape[1]))
        initial = tf.random.uniform(shape, minval=-scale, maxval=scale, dtype=tf.float32)
    elif distribution == 'dl':
        # see page 295 of Goodfellow et al's DL book
        # divide by sqrt of m, where m is number of inputs
        scale = 1.0 / np.sqrt(shape[0])
        initial = tf.random.uniform(shape, minval=-scale, maxval=scale, dtype=tf.float32)
    elif distribution == 'he':
        # from He, et al. ICCV 2015 (referenced in Andrew Ng's class)
        # divide by m, where m is number of inputs
        scale = np.sqrt(2.0 / shape[0])
        initial = tf.random.normal(shape, mean=0, stddev=scale, dtype=tf.float32)
    elif distribution == 'glorot_bengio':
        # see page 295 of Goodfellow et al's DL book
        scale = np.sqrt(6.0 / (shape[0] + shape[1]))
        initial = tf.random.uniform(shape, minval=-scale, maxval=scale, dtype=tf.float32)
    else:
        initial = np.loadtxt(distribution, delimiter=',', dtype=np.float32)
        if (initial.shape[0] != shape[0]) or (initial.shape[1] != shape[1]):
            raise ValueError('Initialization for %s is not correct shape. Expecting (%d,%d), but find (%d,%d) in %s.'
                             % (name, shape[0], shape[1], initial.shape[0], initial.shape[1], distribution))
    return tf.Variable(initial, name=name, trainable=trainable)


def bias_variable(shape, name, distribution, trainable=True):
    """Create a variable for a bias vector.

    Arguments:
        shape -- array giving shape of output bias variable
        var_name -- string naming bias variable
        distribution -- string for which distribution to use for random initialization (file name) (default '')

    Returns:
        a TensorFlow variable for a bias vector
    """

    if distribution == 'uniform':
        initial = tf.random.uniform(shape, minval=-0.2, maxval=0.2, dtype=tf.float32)
    elif distribution == 'normal':
        initial = tf.random.normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32)
    elif distribution == 'none':
        initial = tf.constant(1, shape=shape, dtype=tf.float32)
    else:
        raise NotImplementedError
    return tf.Variable(initial, name=name, trainable=trainable)


def get_dim_list(input_dim, dense_dims, output_dim):
    dim_list = [input_dim]
    for dim in dense_dims:
        dim_list.append(dim)
    dim_list.append(output_dim)
    return dim_list


def exp_length(output_size, epd):
    """Generate shape list of expanded layer.

    Arguments:
        output_size -- [input dimension, layer output size list]
        epd         -- layer expansion order list
    Returns:
        shape list of expanded layer
    """

    layer_shape = np.zeros((len(epd), 2))  # layer shape width
    for layer_index in range(len(output_size) - 1):
        expansion_index = np.ones([output_size[layer_index], 1])  # expansion index
        EP_length = np.sum(expansion_index)  # expansion length
        if epd[layer_index] >= 1:
            for ed in range(epd[layer_index]):
                for g in range(output_size[layer_index]):
                    expansion_index[g] = np.sum(expansion_index[g:(output_size[layer_index])])
                EP_length = np.sum(expansion_index) + EP_length
        layer_shape[layer_index, 0] = output_size[layer_index + 1]
        layer_shape[layer_index, 1] = EP_length
    return layer_shape


def get_knowledge_matrix():
    pass
