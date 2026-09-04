# Witness for wala/ML#874: tf.ones_like's result has the input's shape, and its
# dtype defaults to the input's but is OVERRIDDEN by an explicit dtype=. The old
# pass_through model returned the input, coincidentally right when dtype was
# omitted or matched but confidently wrong when the override differed. Mirrors
# tf.zeros_like, which is modeled correctly.
import tensorflow as tf


def src():
    return tf.ones([3, 4], dtype=tf.float32)


def ones_like_default():
    # dtype omitted: inherits the input's float32.
    return tf.ones_like(src())


def ones_like_override():
    # dtype override DIFFERS from the input's float32: int32 at runtime. The
    # load-bearing arm; pass_through gave float32 here.
    return tf.ones_like(src(), dtype=tf.int32)


assert ones_like_default().dtype == tf.float32
assert ones_like_override().dtype == tf.int32
assert ones_like_default().shape == (3, 4)
