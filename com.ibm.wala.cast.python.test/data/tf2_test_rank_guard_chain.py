# A chain of 160 matrix products inside one function, each input reshaped under a guard on its
# rank that never holds. Each guard reads the rank of the product before it.
import tensorflow as tf


def consume(t):
    assert t.shape == (2, 4)
    assert t.dtype == tf.float32


def chain(x, w):
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    if x.shape.rank == 3:
        x = tf.reshape(x, [-1, 4])
    x = tf.matmul(x, w)
    return x


consume(chain(tf.ones([2, 4]), tf.ones([4, 4])))
