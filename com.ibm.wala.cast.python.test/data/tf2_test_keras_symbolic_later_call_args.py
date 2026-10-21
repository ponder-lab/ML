# Test for wala/ML#1134: applying a program-defined Keras layer to Keras symbolic tensors runs its
# `call` on graph tensors in every argument, not only the first, whether the argument is positional
# or a keyword, and the application's result is a symbolic tensor.
import tensorflow as tf

SEEN = []


def second_inside(t):
    SEEN.append(type(t).__name__)
    return t * 2.0


def keyword_inside(t):
    SEEN.append(type(t).__name__)
    return t * 2.0


class Pair(tf.keras.layers.Layer):

    def call(self, first, second):
        return first + second_inside(second)


class Keyed(tf.keras.layers.Layer):

    def call(self, inputs, other=None):
        return inputs + keyword_inside(other)


def after_pair(x):
    assert type(x).__name__ == "KerasTensor"
    return x + 1.0


def after_keyed(x):
    assert type(x).__name__ == "KerasTensor"
    return x + 1.0


inp = tf.keras.layers.Input([4])
pair = after_pair(Pair()(inp, inp))
keyed = after_keyed(Keyed()(inp, other=inp))
assert SEEN == ["Tensor", "Tensor"]
