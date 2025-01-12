import tensorflow as tf
import numpy as np
import dataclasses
from dataclasses import field
import os


@dataclasses.dataclass
class MlpConfig:
    epochs: int = 10
    evaluation_period: int = 2
    learning_rate: float = 0.001
    model_name: str = "mlp_model"
    hidden_dim: list = field(default_factory=lambda: [128, 128, 128])
    activation: list = field(default_factory=lambda: ["relu", "relu", "relu"])
    # this is the default initializer in tf.keras.layers.Dense,
    # it might be better to use the other initializers when the model doesn't learn
    initializer: str = "glorot_uniform"
    loss_function: str = "mean_squared_error"  # this is the default loss function in tf.keras.losses
    model_path: str = ""
    output_activation: str = "linear"


class MLPBlock(tf.keras.layers.Layer):
    def __init__(self, hidden_dim, activation, initializer):
        super(MLPBlock, self).__init__()
        self.hidden_layers = []
        for i in range(len(hidden_dim)):
            layer = tf.keras.layers.Dense(hidden_dim[i], activation=activation[i],
                                          kernel_initializer=initializer, name=f"dense_{i}")
            self.hidden_layers.append(layer)

    def call(self, inputs, training=None, mask=None):
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)
        return x


class MlpModel:
    def __init__(self, params: MlpConfig, input_shape, output_shape):
        self.params = params
        self.input_shape = input_shape
        self._inputs = tf.keras.layers.Input(shape=(input_shape,), name='input')
        self._mlp_block = MLPBlock(self.params.hidden_dim, self.params.activation, self.params.initializer)
        self._output = tf.keras.layers.Dense(output_shape, activation=self.params.output_activation, name='output')

        self._loss = self._build_loss_function()
        self._optimizer = tf.keras.optimizers.Adam(learning_rate=self.params.learning_rate)
        self.model = self._build_model()
        if self.params.model_path != '':
            self.model.load_weights(self.params.model_path)

    def _build_loss_function(self):
        if self.params.loss_function == "mean_squared_error":
            return tf.keras.losses.MeanSquaredError()
        else:
            raise NotImplementedError

    def _build_model(self):
        x = self._mlp_block(self._inputs)
        output = self._output(x)
        model = tf.keras.Model(inputs=self._inputs, outputs=output, name=self.params.model_name)
        return model

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
        if not os.path.exists(path):
            os.makedirs(path)
        self.model.save_weights(os.path.join(path, "model.weights.h5"))
