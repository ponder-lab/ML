# Test https://github.com/wala/ML/issues/1009: an `ndarray.reshape` result read through a tuple's
# element rather than as the call's own result, as `x_train, x_test = x_train.reshape(...),
# x_test.reshape(...)` does, is typed by the reshape, including where an operation reads it through
# its points-to set, as the division and the dataset built from it do.
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

num_features = 784


def consume(x):
    return x


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
x_train, x_test = x_train.reshape([-1, num_features]), x_test.reshape(
    [-1, num_features]
)
x_train, x_test = x_train / 255.0, x_test / 255.0
assert x_train.shape == (60000, 784) and x_train.dtype == np.float32
train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(256)
for batch_x, batch_y in train.take(1):
    assert batch_x.shape == (256, 784) and batch_x.dtype == tf.float32
    consume(batch_x)
