# Test for wala/ML#1110: TensorFlow 2 runs a module's own body eagerly, so a branch there on
# `tf.executing_eagerly()` takes its `True` arm, and a value only the other arm assigns does not
# reach the merge. Inside a function the call may run under `tf.function`, so the branch is kept.
import tensorflow as tf


def merged(x):
    return tf.abs(x)


def kept(x):
    return tf.abs(x)


def pick_in_function(eager, symbolic):
    if tf.executing_eagerly():
        return eager
    return symbolic


input_tensor = tf.keras.layers.Input([4])

if tf.executing_eagerly():
    pick = tf.ones((2, 4))
else:
    pick = input_tensor

assert merged(pick).shape == (2, 4)

chosen = pick_in_function(tf.ones((2, 4)), input_tensor)
assert kept(chosen).shape == (2, 4)
