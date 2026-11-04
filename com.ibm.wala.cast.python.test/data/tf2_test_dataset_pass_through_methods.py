# Test https://github.com/wala/ML/issues/1214: `skip`, `shard` and `cache` hand on their receiver's
# element, and a batch of one counts the elements it keeps.
import random

import tensorflow as tf


def consume_skipped(x):
    pass


def consume_sharded(x):
    pass


def consume_cached(x):
    pass


def consume_skipped_pair_x(x):
    pass


def consume_skipped_pair_y(y):
    pass


def consume_skipped_batched(x):
    pass


def consume_sharded_batched(x):
    pass


def consume_uneven_shard_batched(x):
    pass


def consume_cached_batched(x):
    pass


def consume_chained_batched(x):
    pass


def consume_unknown_skip_batched(x):
    pass


flat = tf.data.Dataset.from_tensor_slices(tf.ones([8, 4]))
pairs = tf.data.Dataset.from_tensor_slices(
    (tf.ones([8, 4]), tf.zeros([8, 3], tf.int32))
)

for x in flat.skip(1):
    assert x.shape == (4,) and x.dtype == tf.float32
    consume_skipped(x)
for x in flat.shard(2, 0):
    assert x.shape == (4,) and x.dtype == tf.float32
    consume_sharded(x)
for x in flat.cache():
    assert x.shape == (4,) and x.dtype == tf.float32
    consume_cached(x)
for x, y in pairs.skip(1):
    assert x.shape == (4,) and x.dtype == tf.float32
    assert y.shape == (3,) and y.dtype == tf.int32
    consume_skipped_pair_x(x)
    consume_skipped_pair_y(y)

# 8 elements less 1 batch by 2 into three full batches and one of 1.
for x in flat.skip(1).batch(2):
    assert x.shape in ((2, 4), (1, 4))
    consume_skipped_batched(x)
# Every second of 8 elements from the first is 4, batched in full.
for x in flat.shard(2, 0).batch(2):
    assert x.shape == (2, 4)
    consume_sharded_batched(x)
# Every third of 8 elements from the second is 3: one full batch and one of 1.
for x in flat.shard(3, 1).batch(2):
    assert x.shape in ((2, 4), (1, 4))
    consume_uneven_shard_batched(x)
for x in flat.cache().batch(2):
    assert x.shape == (2, 4)
    consume_cached_batched(x)
# A pass-through before the skip keeps the count: 8 less 2 is 6, batched in full.
for x in flat.shuffle(4).skip(2).batch(2):
    assert x.shape == (2, 4)
    consume_chained_batched(x)
# A count the analysis cannot resolve leaves the size unknown.
for x in flat.skip(random.randint(0, 3)).batch(2):
    assert x.shape[1:] == (4,)
    consume_unknown_skip_batched(x)
