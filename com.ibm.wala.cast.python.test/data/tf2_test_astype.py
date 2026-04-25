import tensorflow as tf
import numpy as np


def consume(tensor):
    pass


(x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
assert x_train.shape == (60000, 28, 28)
assert x_train.dtype == np.uint8

y = x_train.astype(np.float32)
assert y.shape == (60000, 28, 28)
assert y.dtype == np.float32

consume(y)
