import tensorflow as tf


def check_rt(rt):
    pass


flat_values = ("a", "b", "c", "d", "e", "f", "g")
nested_value_rowids = (
    (0, 0, 1, 2),
    (0, 1, 1, 2, 2, 2, 3),
)

rt = tf.RaggedTensor.from_nested_value_rowids(flat_values, nested_value_rowids)
assert isinstance(rt, tf.RaggedTensor)
assert rt.shape.as_list() == [3, None, None]
assert rt.dtype == tf.string
check_rt(rt)
