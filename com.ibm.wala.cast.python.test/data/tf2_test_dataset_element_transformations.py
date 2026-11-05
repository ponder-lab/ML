# Test https://github.com/wala/ML/issues/1214: `unbatch`, `apply`, `rejection_resample` and
# `bucket_by_sequence_length` give their datasets an element.
import tensorflow as tf


def consume_unbatched(x):
    pass


def consume_unbatched_scalar(x):
    pass


def consume_unbatched_pair_x(x):
    pass


def consume_unbatched_pair_y(y):
    pass


def consume_unbatched_rows(x):
    pass


def consume_rebatched(x):
    pass


def consume_applied_batch(x):
    pass


def consume_applied(x):
    pass


def consume_resampled_class(c):
    pass


def consume_resampled(x):
    pass


def consume_bucketed(x):
    pass


def consume_resampled_pair_x(x):
    pass


def consume_bucketed_unpadded(x):
    pass


def consume_bucketed_padded(x):
    pass


def consume_bucketed_mixed(x):
    pass


flat = tf.data.Dataset.from_tensor_slices(tf.ones([8, 4]))
pairs = tf.data.Dataset.from_tensor_slices(
    (tf.ones([8, 4]), tf.zeros([8, 3], tf.int32))
)

for x in flat.batch(2).unbatch():
    assert x.shape == (4,) and x.dtype == tf.float32
    consume_unbatched(x)
for x, y in pairs.batch(2).unbatch():
    assert x.shape == (4,) and x.dtype == tf.float32
    assert y.shape == (3,) and y.dtype == tf.int32
    consume_unbatched_pair_x(x)
    consume_unbatched_pair_y(y)
# A dataset of matrices unbatches into their rows.
for x in tf.data.Dataset.from_tensor_slices(tf.ones([8, 2, 4])).unbatch():
    assert x.shape == (4,) and x.dtype == tf.float32
    consume_unbatched_rows(x)
# A scalar has no rows: `unbatch` rejects it.
try:
    for x in tf.data.Dataset.from_tensor_slices(tf.ones([8])).unbatch():
        consume_unbatched_scalar(x)
    assert False
except ValueError:
    pass
# 8 rows batch by 3 into two full batches and one of 2.
for x in flat.batch(2).unbatch().batch(3):
    assert x.shape in ((3, 4), (2, 4))
    consume_rebatched(x)
for x in flat.apply(lambda d: d.skip(1)):
    assert x.shape == (4,) and x.dtype == tf.float32
    consume_applied(x)
for x in flat.apply(lambda d: d.batch(2)):
    assert x.shape == (2, 4) and x.dtype == tf.float32
    consume_applied_batch(x)
for c, x in flat.rejection_resample(
    class_func=lambda v: tf.constant(0, tf.int32),
    target_dist=[1.0, 0.0],
    initial_dist=[1.0, 0.0],
    seed=0,
):
    assert c.shape == () and c.dtype == tf.int32
    assert x.shape == (4,) and x.dtype == tf.float32
    consume_resampled_class(c)
    consume_resampled(x)
for c, (x, y) in pairs.rejection_resample(
    class_func=lambda v, w: tf.constant(0, tf.int32),
    target_dist=[1.0, 0.0],
    initial_dist=[1.0, 0.0],
    seed=0,
):
    assert x.shape == (4,) and x.dtype == tf.float32
    consume_resampled_pair_x(x)
for x in flat.bucket_by_sequence_length(
    element_length_func=lambda v: tf.shape(v)[0],
    bucket_boundaries=[3],
    bucket_batch_sizes=[2, 2],
):
    assert x.shape == (2, 4) and x.dtype == tf.float32
    consume_bucketed(x)
for x in flat.bucket_by_sequence_length(
    element_length_func=lambda v: tf.shape(v)[0],
    bucket_boundaries=[3],
    bucket_batch_sizes=[2, 2],
    pad_to_bucket_boundary=False,
):
    assert x.shape == (2, 4) and x.dtype == tf.float32
    consume_bucketed_unpadded(x)
# A declared padded shape pads each element to it.
for x in flat.bucket_by_sequence_length(
    element_length_func=lambda v: tf.shape(v)[0],
    bucket_boundaries=[3],
    bucket_batch_sizes=[2, 2],
    padded_shapes=[5],
):
    assert x.shape == (2, 5) and x.dtype == tf.float32
    consume_bucketed_padded(x)

# Elements whose shapes differ on two axes share a batch, padded on each axis to its longest.
mixed = tf.data.Dataset.from_tensor_slices(tf.ones([8, 2, 5])).concatenate(
    tf.data.Dataset.from_tensor_slices(tf.ones([8, 4, 3]))
)
for x in mixed.bucket_by_sequence_length(
    element_length_func=lambda v: tf.shape(v)[0],
    bucket_boundaries=[10],
    bucket_batch_sizes=[3, 3],
):
    assert x.shape[1:] in ((2, 5), (4, 5), (4, 3)) and x.dtype == tf.float32
    consume_bucketed_mixed(x)
