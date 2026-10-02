# https://github.com/wala/ML/issues/1009: a function nested deeper than the call-string depth
# merges its callers into one node, so its parameter holds every caller's argument.

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist


def consume(y):
    return y


def project(x):
    y = tf.keras.layers.Dense(4)(x)
    consume(y)
    return y


def relay4(x):
    return project(x)


def relay3(x):
    return relay4(x)


def relay2(x):
    return relay3(x)


def relay1(x):
    return relay2(x)


def relay(x):
    return relay1(x)


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
x_train, x_test = x_train.reshape([-1, 784]), x_test.reshape([-1, 784])

for batch_x in tf.data.Dataset.from_tensor_slices(x_train).batch(256).take(1):
    assert batch_x.shape == (256, 784)
    assert relay(batch_x).shape == (256, 4)

test_images = x_test[:5]
assert relay(test_images).shape == (5, 4)


def consume_both(y):
    return y


def project_both(x):
    y = tf.keras.layers.Dense(4)(x)
    consume_both(y)
    return y


def relay_both4(x):
    return project_both(x)


def relay_both3(x):
    return relay_both4(x)


def relay_both2(x):
    return relay_both3(x)


def relay_both1(x):
    return relay_both2(x)


def relay_both(x):
    return relay_both1(x)


assert relay_both(x_test[:5]).shape == (5, 4)
assert relay_both(x_test[:7]).shape == (7, 4)
