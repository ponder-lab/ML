# Probe for the wala/ML#570 residual: a list accumulated with `append` in a loop feeds
# `tf.concat`, mirroring `MessagePassing._calculate_messages_all_type` feeding
# `_aggregate_function`. The appended values' shapes and dtype survive the list: the concat
# result keeps the rank and non-axis dims with a dynamic axis dim (the element count is not
# statically known) and the float32 dtype.
import tensorflow as tf


def consume(t):
    pass


xs = []
for i in range(3):
    xs.append(tf.ones((2, 8)))

y = tf.concat(xs, axis=0)
consume(y)

assert y.shape == (6, 8)
assert y.dtype == tf.float32
