# Test for wala/ML#1109: applying a program-defined Keras layer to a Keras symbolic tensor runs its
# `call` on a graph tensor, not on the symbolic one, and the application's result is the symbolic
# tensor, whatever the body computes with.
import tensorflow as tf

SEEN = []


def scale_inside(t):
    SEEN.append(type(t).__name__)
    return t * 2.0


class Scaled(tf.keras.layers.Layer):

    def call(self, inputs):
        return scale_inside(inputs)


class Relu(tf.keras.layers.Layer):

    def call(self, inputs):
        return tf.nn.relu(inputs)


def after_scaled(x):
    return x + 1.0


def after_relu(x):
    return x + 1.0


inp = tf.keras.layers.Input([16, 16, 3])
scaled = after_scaled(Scaled()(inp))
relu = after_relu(Relu()(inp))
assert type(scaled).__name__ == "KerasTensor"
assert type(relu).__name__ == "KerasTensor"
assert SEEN == ["Tensor"]
