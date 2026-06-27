import tensorflow as tf


def check_input(i1):
    pass


# `type_spec` supplies the full type: the result takes the spec's shape and dtype verbatim, with no
# batch dimension prepended. See https://github.com/wala/ML/issues/617.
spec = tf.TensorSpec(shape=(None, 4), dtype=tf.int32)
input1 = tf.keras.Input(type_spec=spec)
assert input1.shape == (None, 4)
assert input1.dtype == tf.int32

check_input(input1)
