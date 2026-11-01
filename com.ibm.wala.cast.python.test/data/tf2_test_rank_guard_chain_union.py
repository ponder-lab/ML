# A value that is a constant on one path and the end of a chain of three rank-guarded matrix
# products on the other. A read that cuts the chain keeps the constant beside an unknown member.
import os

import tensorflow as tf


def consume(t):
    assert t.dtype == tf.float32
    assert t.shape in [(3, 3), (2, 4)]


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
    return x


def pick(x, w):
    if os.environ.get("PICK_CONSTANT") is not None:
        return tf.ones([3, 3])
    return chain(x, w)


consume(pick(tf.ones([2, 4]), tf.ones([4, 4])))
