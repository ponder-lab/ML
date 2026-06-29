import tensorflow as tf


def consume(x):
    pass


# wala/ML#637's example is `tf.constant([1 + 2j, 3 + 4j], dtype=tf.complex64)`, but a complex
# literal (`2j`) currently breaks call-graph entrypoint creation (the separate front-end gap
# wala/ML#642), so a faithful copy would not build at all. Use integer values cast to `complex64` to
# isolate the dtype modeling; the dtype comes from the explicit `dtype=` argument either way.
z = tf.constant([1, 2], dtype=tf.complex64)
assert z.shape == (2,)
assert z.dtype == tf.complex64
consume(z)
