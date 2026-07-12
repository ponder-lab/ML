import os
import tensorflow as tf


def consume(t):
    pass


# A configuration-sourced size: fixed for any given run, but the static
# analysis cannot compute it (the value crosses an environment read). The
# resulting axis must type as *unresolved* (a fixed size of unknown value),
# not *dynamic* (a `None` axis in the runtime `TensorShape`) — wala/ML#721.
n = int(os.environ.get("ARIADNE_TEST_N", "4"))

x = tf.ones((n, 100))

assert x.shape == (4, 100)
assert x.dtype == tf.float32

consume(x)
