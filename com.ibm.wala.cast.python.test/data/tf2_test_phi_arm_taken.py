import tensorflow as tf


def consume(x):
    pass


def f(x):
    # Two-arm phi at consume(x): the skip edge leaves the branch block directly (decidable), the
    # expand edge sits behind the expand_dims invoke split (undecidable). Called rank-2 and rank-3
    # so 1-CFA splits; in the rank-3 context the skip arm is decidably taken and the expand arm is
    # merely undecidable (wala/ML#902).
    if x.shape.ndims == 2:
        x = tf.expand_dims(x, -1)
    consume(x)


f(tf.ones((16, 100)))
f(tf.ones((8, 10, 4)))
