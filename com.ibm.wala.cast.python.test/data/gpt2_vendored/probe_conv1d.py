# Probe driver for wala/ML#618: the vendored `Conv1d` forward output in isolation.
# Analyzed statically, like `A.py` itself.
import tensorflow as tf

from layers.feed_forward import Conv1d


def consume(t):
    pass


c = Conv1d(8, 16)
out = c(tf.ones((2, 3, 8)))
# Observed by running this file under `python3.10` (wala/ML#808). The analysis pins this result at
# unknown rank because the reshape's target shape is a runtime-built list (wala/ML#703); the
# observation is what that gap costs, and the concrete answer closing it should reach.
assert out.shape == (2, 3, 16) and out.dtype == tf.float32
consume(out)
