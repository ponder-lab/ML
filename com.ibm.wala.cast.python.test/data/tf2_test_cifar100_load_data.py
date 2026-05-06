# Test of tf.keras.datasets.cifar100.load_data(). Same shapes and dtype as cifar10:
# x_train (50000, 32, 32, 3) uint8, y_train (50000, 1) uint8, x_test (10000, 32, 32, 3) uint8, y_test (10000, 1) uint8.
import tensorflow as tf


def f(a):
    pass


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
assert x_train.shape == (50000, 32, 32, 3)
assert x_test.shape == (10000, 32, 32, 3)
assert x_train.dtype == "uint8"
f(x_train)
