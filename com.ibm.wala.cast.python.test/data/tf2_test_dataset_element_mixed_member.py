import random

import tensorflow as tf


def consume(t):
    assert isinstance(t, tf.Tensor)
    assert t.shape == (3,)
    assert t.dtype == tf.float32


# The subscript is `0` or the name `"foo"` to the analysis, which cannot fold the condition; at run
# time it is always `0`. The index is the element's component, and the name selects none.
key = 0 if random.random() < 2 else "foo"

dataset = tf.data.Dataset.from_tensor_slices(
    (tf.ones([10, 3]), tf.zeros([10], tf.int32))
)

for element in dataset:
    consume(element[key])
