# Test of tf.keras.datasets.fashion_mnist.load_data(). Same shapes and dtype as mnist:
# x_train (60000, 28, 28) uint8, y_train (60000,) uint8, x_test (10000, 28, 28) uint8, y_test (10000,) uint8.
import tensorflow as tf


def f(a):
    pass


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
assert x_train.shape == (60000, 28, 28)
assert y_train.shape == (60000,)
assert x_test.shape == (10000, 28, 28)
assert y_test.shape == (10000,)
assert x_train.dtype == "uint8"
f(x_train)
