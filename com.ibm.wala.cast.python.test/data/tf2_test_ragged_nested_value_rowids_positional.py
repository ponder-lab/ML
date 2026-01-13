import tensorflow as tf

flat_values = tf.constant(["a", "b", "c", "d", "e", "f", "g"])
nested_value_rowids = [
    tf.constant([0, 0, 1, 2], dtype=tf.int64),
    tf.constant([0, 1, 1, 2, 2, 2, 3], dtype=tf.int64),
]

rt = tf.RaggedTensor.from_nested_value_rowids(flat_values, nested_value_rowids)
assert isinstance(rt, tf.RaggedTensor)
assert rt.shape.as_list() == [3, None, None]
assert rt.dtype == tf.string
