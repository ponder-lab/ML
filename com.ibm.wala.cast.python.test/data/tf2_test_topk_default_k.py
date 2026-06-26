import tensorflow as tf


def consume(v):
    pass


# `tf.math.top_k` with `k` omitted defaults to `k=1`, so `values` of a `(4,)`
# input is `(1,)`. Exercises the `k`-default path of the wala/ML#609 composer.
values, indices = tf.math.top_k(tf.constant([1.0, 3.0, 2.0, 5.0]))
consume(values)
