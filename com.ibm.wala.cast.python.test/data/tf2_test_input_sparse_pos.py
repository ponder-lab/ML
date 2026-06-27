import tensorflow as tf


def check_input(i1):
    pass


# Input(shape, batch_size, name, dtype, sparse); sparse is the 5th arg (index 4).
# A sparse `Input` has the same logical shape and dtype as a dense one; `sparse` only affects
# storage. See https://github.com/wala/ML/issues/616.
input1 = tf.keras.Input((10,), None, None, None, True)
assert input1.shape == (None, 10)
assert input1.dtype == tf.float32

check_input(input1)
