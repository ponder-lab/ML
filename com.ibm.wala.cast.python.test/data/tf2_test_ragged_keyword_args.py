from tensorflow.python.ops.ragged.ragged_tensor import RaggedTensor
import tensorflow as tf


def test_keywords(rt1, rt2, rt3, rt4):
    pass


# 1. Standard keyword order
rt1 = RaggedTensor.from_value_rowids(
    value_rowids=[0, 0, 1, 1, 1], values=[1, 2, 3, 4, 5]
)
assert rt1.shape.as_list() == [2, None]
assert rt1.dtype == tf.int32

# 2. Reversed keyword order
rt2 = RaggedTensor.from_value_rowids(
    values=[1, 2, 3, 4, 5], value_rowids=[0, 0, 1, 1, 1]
)
assert rt2.shape.as_list() == [2, None]
assert rt2.dtype == tf.int32

# 3. Mix of positional and keyword
rt3 = RaggedTensor.from_value_rowids([1, 2, 3, 4, 5], value_rowids=[0, 0, 1, 1, 1])
assert rt3.shape.as_list() == [2, None]
assert rt3.dtype == tf.int32

# 4. Explicit nrows as keyword
rt4 = RaggedTensor.from_value_rowids(
    value_rowids=[0, 0, 1, 1, 1], values=[1, 2, 3, 4, 5], nrows=3
)
assert rt4.shape.as_list() == [3, None]
assert rt4.dtype == tf.int32

test_keywords(rt1, rt2, rt3, rt4)
