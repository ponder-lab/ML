# Probe for the collection-dataflow family (wala/ML#570/#618): a tensor read back by iterating a
# LIST LITERAL keeps its type (differential probe against the `append` and `zip` forms).
import tensorflow as tf


def consume(t):
    pass


tensors = [tf.ones((4, 8)), tf.ones((4, 8))]

for t in tensors:
    consume(t)
    assert t.shape == (4, 8)
    assert t.dtype == tf.float32
