import tensorflow as tf


def consume(x):
    pass


# `tf.math.top_k` returns a NamedTuple whose object catalog carries string field-alias keys
# (`values`, `indices`) alongside the integer element indices `0`, `1`. Slicing it makes the
# shape/dtype catalog walk hit those non-integer keys. Regression guard for wala/ML#603: the
# non-integer keys must be filtered (not crash `getFieldIndex`); the slice then recovers the
# element dtypes (float32 values, int32 indices). The ⊤ shape is the unmodeled top_k
# output shape, tracked by wala/ML#609. No `read_data` is involved, which
# is why wala/ML#380 would not fix this case.
r = tf.math.top_k(tf.constant([1.0, 3.0, 2.0, 5.0]), k=2)
consume(r[0:1])
