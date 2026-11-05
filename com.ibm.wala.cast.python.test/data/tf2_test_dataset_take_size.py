# Test https://github.com/wala/ML/issues/1216: `take(count)` has the least of `count` and its
# receiver's size, so a batch of it counts the elements it keeps.
import random

import tensorflow as tf


def consume_taken_batched(x):
    pass


def consume_all_taken_batched(x):
    pass


def consume_over_taken_batched(x):
    pass


def consume_unknown_take_batched(x):
    pass


def consume_taken_pair_x(x):
    pass


def consume_repeated_taken_batched(x):
    pass


flat = tf.data.Dataset.from_tensor_slices(tf.ones([8, 4]))
pairs = tf.data.Dataset.from_tensor_slices(
    (tf.ones([8, 4]), tf.zeros([8, 3], tf.int32))
)

# Three elements batch by 2 into a full batch and one of 1.
for x in flat.take(3).batch(2):
    assert x.shape in ((2, 4), (1, 4)) and x.dtype == tf.float32
    consume_taken_batched(x)

# A count of -1 takes every element: 8 batch by 3 into two full batches and one of 2.
for x in flat.take(-1).batch(3):
    assert x.shape in ((3, 4), (2, 4)) and x.dtype == tf.float32
    consume_all_taken_batched(x)

# A count past the receiver's size takes every element: 8 batch by 4 into two full batches.
for x in flat.take(10).batch(4):
    assert x.shape == (4, 4) and x.dtype == tf.float32
    consume_over_taken_batched(x)

# A count that does not resolve leaves the size unknown, so a batch may be short.
for x in flat.take(random.randint(1, 8)).batch(2):
    assert x.shape in ((2, 4), (1, 4)) and x.dtype == tf.float32
    consume_unknown_take_batched(x)

for x, y in pairs.take(3).batch(2):
    assert x.shape in ((2, 4), (1, 4)) and x.dtype == tf.float32
    consume_taken_pair_x(x)

# A take of a repeated dataset is finite: 5 elements batch by 2 into two full batches and one of 1.
for x in flat.repeat().take(5).batch(2):
    assert x.shape in ((2, 4), (1, 4)) and x.dtype == tf.float32
    consume_repeated_taken_batched(x)
