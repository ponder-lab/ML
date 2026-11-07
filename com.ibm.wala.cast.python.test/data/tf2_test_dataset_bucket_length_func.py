# `bucket_by_sequence_length` calls its `element_length_func` on the receiver's element, the method
# and `tf.data.experimental.bucket_by_sequence_length` handed to `apply` alike, and spreads a tuple
# element into its arguments (wala/ML#1214).
import tensorflow as tf


def consume_length_arg(v):
    pass


def consume_applied_length_arg(v):
    pass


def consume_pair_length_x(x):
    pass


def consume_pair_length_y(y):
    pass


def length_of(v):
    assert v.shape == (4,) and v.dtype == tf.float32, v
    consume_length_arg(v)
    return tf.shape(v)[0]


def applied_length_of(v):
    assert v.shape == (4,) and v.dtype == tf.float32, v
    consume_applied_length_arg(v)
    return tf.shape(v)[0]


def pair_length_of(x, y):
    assert x.shape == (4,) and x.dtype == tf.float32, x
    assert y.shape == (3,) and y.dtype == tf.int32, y
    consume_pair_length_x(x)
    consume_pair_length_y(y)
    return tf.shape(x)[0]


flat = tf.data.Dataset.from_tensor_slices(tf.ones([8, 4]))

for x in flat.bucket_by_sequence_length(
    element_length_func=length_of,
    bucket_boundaries=[3],
    bucket_batch_sizes=[2, 2],
):
    pass

for x in flat.apply(
    tf.data.experimental.bucket_by_sequence_length(
        element_length_func=applied_length_of,
        bucket_boundaries=[3],
        bucket_batch_sizes=[2, 2],
    )
):
    pass

pairs = tf.data.Dataset.from_tensor_slices(
    (tf.ones([8, 4]), tf.zeros([8, 3], tf.int32))
)
for x, y in pairs.bucket_by_sequence_length(
    element_length_func=pair_length_of,
    bucket_boundaries=[3],
    bucket_batch_sizes=[2, 2],
):
    pass
