import os
import tensorflow as tf


def consume(t):
    pass


# `tf.range` with a configuration-sourced limit: the bound is a Python scalar
# fixed for any given run, but the analysis cannot compute it (the value
# crosses an environment read), so the rank-1 length must type as *unresolved*
# rather than *dynamic* (wala/ML#721).
limit = int(os.environ.get("ARIADNE_TEST_LIMIT", "5"))

x = tf.range(limit)

assert x.shape == (5,)
assert x.dtype == tf.int32

consume(x)
