import tensorflow as tf


def check_rt(rt):
    pass


flat_values = (1, 2, 3)
nested_value_rowids = (
    (0, 0, 1),
    (0, 1, 2),
)

rt = tf.RaggedTensor.from_nested_value_rowids(
    flat_values=flat_values, nested_value_rowids=nested_value_rowids
)
assert isinstance(rt, tf.RaggedTensor)
assert rt.shape.as_list() == [2, None, None]
assert rt.dtype == tf.int32
check_rt(rt)
