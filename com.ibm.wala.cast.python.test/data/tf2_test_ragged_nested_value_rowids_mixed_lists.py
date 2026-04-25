import tensorflow as tf


def check_rt(rt):
    pass


flat_values = [1.0, 2.0]
nested_value_rowids = [
    [0],
    [0, 0],
]

rt = tf.RaggedTensor.from_nested_value_rowids(
    flat_values, nested_value_rowids=nested_value_rowids, validate=False
)
assert isinstance(rt, tf.RaggedTensor)
assert rt.shape.as_list() == [1, None, None]
assert rt.dtype == tf.float32
check_rt(rt)
