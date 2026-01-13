import tensorflow as tf

flat_values = tf.constant([1.0, 2.0], dtype=tf.float32)
nested_value_rowids = [
    tf.constant([0], dtype=tf.int64),
    tf.constant([0, 0], dtype=tf.int64),
]

rt = tf.RaggedTensor.from_nested_value_rowids(
    flat_values, nested_value_rowids=nested_value_rowids, validate=False
)
assert isinstance(rt, tf.RaggedTensor)
assert rt.shape.as_list() == [1, None, None]
assert rt.dtype == tf.float32
