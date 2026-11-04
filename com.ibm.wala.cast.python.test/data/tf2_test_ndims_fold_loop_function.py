# The loop of tf2_test_ndims_fold_loop.py in a function called twice, so its guard is read in two
# contexts.
import tensorflow as tf


def consume(x):
    pass


def flatten(x):
    for _ in range(3):
        if x.shape.rank == 2:
            x = tf.reshape(x, [4])
    return x


y = flatten(tf.ones([2, 2]))
assert y.shape == (4,)
consume(y)
z = flatten(tf.ones([2, 2]))
consume(z)
