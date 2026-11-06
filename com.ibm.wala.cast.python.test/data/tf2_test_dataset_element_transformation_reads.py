# Test https://github.com/wala/ML/issues/1214: the members of a bucketed or unbatched element read
# one at a time, the class of a resampled element read by its index, and a batch of a bucketed or
# resampled dataset.
import tensorflow as tf


def consume_bucketed_pair_x(x):
    pass


def consume_bucketed_pair_y(y):
    pass


def consume_bucketed_batch(x):
    pass


def consume_resampled_class_read(c):
    pass


def consume_resampled_batch(x):
    pass


def consume_unbatched_dict(x):
    pass


def consume_resampled_list_class(c):
    pass


def consume_bucketed_mapped_x(x):
    pass


flat = tf.data.Dataset.from_tensor_slices(tf.ones([8, 4]))
pairs = tf.data.Dataset.from_tensor_slices(
    (tf.ones([8, 4]), tf.zeros([8, 3], tf.int32))
)

for x, y in pairs.bucket_by_sequence_length(
    element_length_func=lambda v, w: tf.shape(v)[0],
    bucket_boundaries=[3],
    bucket_batch_sizes=[2, 2],
):
    assert x.shape == (2, 4) and x.dtype == tf.float32
    assert y.shape == (2, 3) and y.dtype == tf.int32
    consume_bucketed_pair_x(x)
    consume_bucketed_pair_y(y)

# Four batches of 2 rows, batched by 2.
bucketed = flat.bucket_by_sequence_length(
    element_length_func=lambda v: tf.shape(v)[0],
    bucket_boundaries=[3],
    bucket_batch_sizes=[2, 2],
)
for x in bucketed.batch(2):
    assert x.shape == (2, 2, 4) and x.dtype == tf.float32
    consume_bucketed_batch(x)

resampled = flat.rejection_resample(
    class_func=lambda v: tf.constant(0, tf.int32),
    target_dist=[1.0, 0.0],
    initial_dist=[1.0, 0.0],
    seed=0,
)
for element in resampled:
    c = element[0]
    assert c.shape == () and c.dtype == tf.int32
    consume_resampled_class_read(c)
for c, x in resampled.batch(2):
    assert x.shape[1:] == (4,) and x.dtype == tf.float32
    consume_resampled_batch(x)

dicts = tf.data.Dataset.from_tensor_slices(
    {"a": tf.ones([8, 4]), "b": tf.zeros([8, 3], tf.int32)}
)
for d in dicts.batch(2).unbatch():
    x = d["a"]
    assert x.shape == (4,) and x.dtype == tf.float32
    consume_unbatched_dict(x)

resampled_list = tf.data.Dataset.from_tensor_slices(
    [(7, 8), (9, 10)]
).rejection_resample(
    class_func=lambda v: tf.constant(0, tf.int32),
    target_dist=[1.0, 0.0],
    initial_dist=[1.0, 0.0],
    seed=0,
)
for element in resampled_list:
    c = element[0]
    assert c.shape == () and c.dtype == tf.int32
    consume_resampled_list_class(c)

for x, y in pairs.map(lambda a, b: (a * 2.0, b)).bucket_by_sequence_length(
    element_length_func=lambda v, w: tf.shape(v)[0],
    bucket_boundaries=[3],
    bucket_batch_sizes=[2, 2],
):
    assert x.shape == (2, 4) and x.dtype == tf.float32
    consume_bucketed_mapped_x(x)
