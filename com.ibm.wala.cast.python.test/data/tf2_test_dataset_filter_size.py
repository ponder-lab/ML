# Test https://github.com/wala/ML/issues/1237: `filter` keeps an uncounted number of its receiver's
# elements, so a batch of it may end with a partial batch, and a filter of a dataset that never
# ends never ends either, so a batch of that has no partial batch.
import tensorflow as tf


def consume_filtered_batched(x):
    pass


def consume_unfiltered_batched(x):
    pass


def consume_repeated_filtered_batched(x):
    pass


def consume_filtered_pair_x(x):
    pass


ranged = tf.data.Dataset.from_tensor_slices(tf.range(4))

# The filter keeps three of the four elements, so the batches are (2,) and (1,).
for x in ranged.filter(lambda v: v < 3).batch(2):
    assert x.shape in ((2,), (1,)) and x.dtype == tf.int32, x
    consume_filtered_batched(x)

# Four elements batch by 2 into two full batches.
for x in ranged.batch(2):
    assert x.shape == (2,) and x.dtype == tf.int32, x
    consume_unfiltered_batched(x)

# A filter of an endless repeat is endless, so every batch is full.
for x in ranged.repeat().filter(lambda v: v < 3).batch(2).take(3):
    assert x.shape == (2,) and x.dtype == tf.int32, x
    consume_repeated_filtered_batched(x)

pairs = tf.data.Dataset.from_tensor_slices((tf.ones([4, 3]), tf.range(4)))

# Both members of a pair keep the receiver's element, and the batches are (2, 3) and (1, 3).
for x, y in pairs.filter(lambda a, b: b < 3).batch(2):
    assert x.shape in ((2, 3), (1, 3)) and x.dtype == tf.float32, x
    consume_filtered_pair_x(x)
