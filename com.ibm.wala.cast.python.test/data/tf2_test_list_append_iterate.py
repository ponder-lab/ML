# Probe for the collection-dataflow family (wala/ML#570): a tensor appended to a list in a loop
# and read back by iteration keeps its type (mirroring `messages_all_type.append(messages)`).
import tensorflow as tf


def consume(t):
    pass


tensors = []
for _ in range(3):
    tensors.append(tf.ones((4, 8)))

for t in tensors:
    consume(t)
    assert t.shape == (4, 8)
    assert t.dtype == tf.float32
