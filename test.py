import tensorflow as tf
import numpy as np
from dataclasses import dataclass, field
from tensorflow.keras.layers import Input, Dense, LayerNormalization, Concatenate
from tensorflow.keras import Model

z = 1.2
tem = 20.0
lam = (tf.math.exp(z * tem) - 1)/ (tf.math.exp(tem) - 1)
print(lam)
