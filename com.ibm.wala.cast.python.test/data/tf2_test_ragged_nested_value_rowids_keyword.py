import tensorflow as tf

flat_values = tf.constant([1, 2, 3], dtype=tf.int32)
nested_value_rowids = [
    tf.constant([0, 0, 1], dtype=tf.int64),
    tf.constant([0, 1, 2], dtype=tf.int64),
]

rt = tf.RaggedTensor.from_nested_value_rowids(
    flat_values=flat_values, nested_value_rowids=nested_value_rowids
)
assert isinstance(rt, tf.RaggedTensor)
assert rt.shape.as_list() == [2, None, None]
assert rt.dtype == tf.int32
