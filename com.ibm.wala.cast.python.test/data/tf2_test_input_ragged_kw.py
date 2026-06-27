import tensorflow as tf


def check_input(i1):
    pass


# A ragged `Input` has the same tracked shape and dtype as a dense one; raggedness only affects the
# tensor's row structure, not its `.shape` or `.dtype`. See https://github.com/wala/ML/issues/617.
input1 = tf.keras.Input(shape=(10,), ragged=True)
assert input1.shape == (None, 10)
assert input1.dtype == tf.float32

check_input(input1)
