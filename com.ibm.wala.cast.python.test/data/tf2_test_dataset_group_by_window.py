# The element of `group_by_window` is the element of the datasets its `reduce_func` returns, and
# `tf.data.experimental.group_by_window` and `bucket_by_sequence_length` handed to `apply` give the
# same element as the methods do (wala/ML#1214).
import tensorflow as tf


def consume_grouped(x):
    pass


def consume_grouped_window_size_func(x):
    pass


def consume_grouped_applied(x):
    pass


def consume_grouped_pair_x(x):
    pass


def consume_grouped_pair_y(y):
    pass


def consume_key(k):
    pass


def consume_bucketed_applied(x):
    pass


flat = tf.data.Dataset.from_tensor_slices(tf.ones([8, 4]))

# Windows of three of eight elements, so a batch of two may hold one or two rows.
for x in flat.group_by_window(
    key_func=lambda v: tf.constant(0, tf.int64),
    reduce_func=lambda k, d: d.batch(2),
    window_size=3,
):
    assert x.shape[1:] == (4,) and x.dtype == tf.float32, x
    consume_grouped(x)

for x in flat.group_by_window(
    key_func=lambda v: tf.constant(0, tf.int64),
    reduce_func=lambda k, d: d.batch(2),
    window_size_func=lambda k: tf.constant(3, tf.int64),
):
    assert x.shape[1:] == (4,) and x.dtype == tf.float32, x
    consume_grouped_window_size_func(x)

for x in flat.apply(
    tf.data.experimental.group_by_window(
        key_func=lambda v: tf.constant(0, tf.int64),
        reduce_func=lambda k, d: d.batch(2),
        window_size=3,
    )
):
    assert x.shape[1:] == (4,) and x.dtype == tf.float32, x
    consume_grouped_applied(x)

pairs = tf.data.Dataset.from_tensor_slices(
    (tf.ones([8, 4]), tf.zeros([8, 3], tf.int32))
)
for x, y in pairs.group_by_window(
    key_func=lambda a, b: tf.constant(0, tf.int64),
    reduce_func=lambda k, d: d.batch(2),
    window_size=3,
):
    assert x.shape[1:] == (4,) and x.dtype == tf.float32, x
    assert y.shape[1:] == (3,) and y.dtype == tf.int32, y
    consume_grouped_pair_x(x)
    consume_grouped_pair_y(y)


def reduce_with_key(key, window):
    assert key.shape == () and key.dtype == tf.int64, key
    consume_key(key)
    return window.batch(2)


for x in flat.group_by_window(
    key_func=lambda v: tf.constant(0, tf.int64),
    reduce_func=reduce_with_key,
    window_size=3,
):
    pass

for x in flat.apply(
    tf.data.experimental.bucket_by_sequence_length(
        element_length_func=lambda v: tf.shape(v)[0],
        bucket_boundaries=[3],
        bucket_batch_sizes=[2, 2],
    )
):
    assert x.shape == (2, 4) and x.dtype == tf.float32, x
    consume_bucketed_applied(x)
