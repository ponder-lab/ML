import tensorflow as tf


def consume_sum(y):
    pass


def consume_max(y):
    pass


def consume_mean(y):
    pass


# `tf.math.unsorted_segment_{sum,max,mean}(data, segment_ids, num_segments)` aggregate
# the rows of `data` that share a `segment_ids` value into `num_segments` rows. The
# output dtype is `data`'s; the leading axis is `num_segments` (a runtime value), so the
# static analysis recovers the dtype but leaves the shape at ⊤ (wala/ML#570).
data = tf.constant(
    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=tf.float32
)
segment_ids = tf.constant([0, 1, 0], dtype=tf.int32)
num_segments = 2

s = tf.math.unsorted_segment_sum(data, segment_ids, num_segments)
assert s.shape == (2, 3) and s.dtype == tf.float32
consume_sum(s)

x = tf.math.unsorted_segment_max(data, segment_ids, num_segments)
assert x.shape == (2, 3) and x.dtype == tf.float32
consume_max(x)

m = tf.math.unsorted_segment_mean(data, segment_ids, num_segments)
assert m.shape == (2, 3) and m.dtype == tf.float32
consume_mean(m)
