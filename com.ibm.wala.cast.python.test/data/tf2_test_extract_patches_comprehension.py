import tensorflow as tf


def f(a):
    pass


n = 10
# `images` built from a nested list comprehension (the wala/ML#584 corpus case).
images = [[[[x * n + y + 1] for y in range(n)] for x in range(n)]]
r = tf.image.extract_patches(
    images=images,
    sizes=[1, 3, 3, 1],
    strides=[1, 5, 5, 1],
    rates=[1, 1, 1, 1],
    padding="VALID",
)
assert r.dtype == tf.int32
f(r)
