# Test for wala/ML#1110: a module body's code inside `with tf.Graph().as_default():` runs in graph
# mode, so a branch there on `tf.executing_eagerly()` takes its `False` arm, and the analysis must keep
# both arms rather than fold the condition to `True`.
import tensorflow as tf


def graph_merged(x):
    return x


input_tensor = tf.keras.layers.Input([4])

with tf.Graph().as_default():
    assert not tf.executing_eagerly()
    if tf.executing_eagerly():
        pick = tf.ones((2, 4))
    else:
        pick = input_tensor

    assert graph_merged(pick) is input_tensor
