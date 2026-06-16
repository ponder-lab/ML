import tensorflow as tf


def f(a):
    pass


images = tf.random.uniform((1, 10, 10, 3))
y = tf.image.extract_patches(
    images=images,
    sizes=[1, 3, 3, 1],
    strides=[1, 5, 5, 1],
    rates=[1, 1, 1, 1],
    padding="VALID",
)
assert isinstance(y, tf.Tensor)
# VALID padding: out = (10 - 3) // 5 + 1 = 2 per spatial axis; depth = 3 * 3 * 3 = 27.
assert y.shape == (1, 2, 2, 27)
assert y.dtype == tf.float32

f(y)
