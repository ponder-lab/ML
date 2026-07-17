import tensorflow as tf


def combine(a, b, c, scale=1):
    return tf.reduce_sum(a) * scale


def driver(rest):
    for i in range(3):
        combine(tf.constant([1.0, 2.0]), *rest, i)


driver([tf.constant([3.0, 4.0]), tf.constant([5.0, 6.0])])
