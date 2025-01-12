from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras import Model
import tensorflow as tf


LOG_STD_MAX = 2
LOG_STD_MIN = -5

def build_mlp_model_deterministic(obs_input_shape, obs_aug_shape, shape_output, num_context, name='', output_activation=None):
    # for context prediction
    # joint
    input_layer = Input(shape=(obs_aug_shape + obs_input_shape,), name=name + 'input_layer', dtype=tf.float16)
    con_dense1 = Dense(128, activation='relu', name=name + 'con_dense1')(input_layer[:, :obs_aug_shape])
    con_dense2 = Dense(128, activation='relu', name=name + 'con_dense2')(con_dense1)
    con_dense3 = Dense(128, activation='relu', name=name + 'con_dense3')(con_dense2)
    context_output = Dense(num_context, activation='relu', name=name + 'con_out')(con_dense3)

    input_concat = Concatenate()([input_layer[:, obs_aug_shape:], context_output])
    dense1 = Dense(128, activation='relu', name=name + 'dense1')(input_concat)
    dense2 = Dense(128, activation='relu', name=name + 'dense2')(dense1)
    dense3 = Dense(128, activation='relu', name=name + 'dense3')(dense2)
    output = Dense(shape_output, activation=output_activation, name=name + 'output')(dense3)
    model = Model(inputs=input_layer, outputs=output, name=name)
    return model

def build_mlp_model_gaussian(obs_input_shape, obs_aug_shape, shape_output, num_context, name='', output_activation=None):
    # for context prediction

    input_layer = Input(shape=(obs_aug_shape + obs_input_shape,), name=name + 'input_layer', dtype=tf.float16)
    con_dense1 = Dense(128, activation='relu', name=name + 'con_dense1')(input_layer[:, :obs_aug_shape])
    con_dense2 = Dense(128, activation='relu', name=name + 'con_dense2')(con_dense1)
    con_dense3 = Dense(128, activation='relu', name=name + 'con_dense3')(con_dense2)
    context_output = Dense(num_context, activation='relu', name=name + 'con_out')(con_dense3)

    input_concat = Concatenate()([input_layer[:, obs_aug_shape:], context_output])
    dense1 = Dense(128, activation='relu', name=name + 'dense1')(input_concat)
    dense2 = Dense(128, activation='relu', name=name + 'dense2')(dense1)
    dense3 = Dense(128, activation='relu', name=name + 'dense3')(dense2)
    mean = Dense(shape_output, activation=None, name=name + 'mean')(dense3)
    log_std = Dense(shape_output, activation='tanh', name=name + 'std')(dense3)
    # rescaling into interval [LOG_STD_MIN, LOG_STD_MAX] using first order taylor approximation
    log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
    model = Model(inputs=input_layer, outputs=[mean, log_std])
    return model