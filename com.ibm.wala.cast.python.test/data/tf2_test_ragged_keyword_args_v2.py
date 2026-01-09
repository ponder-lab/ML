from tensorflow.python.ops.ragged.ragged_tensor import RaggedTensor
import tensorflow as tf


def test_keywords(rt1, rt2, rt3, rt4):
    pass


# 1. Standard keyword order (matching documentation: values, then value_rowids)
rt1 = RaggedTensor.from_value_rowids(
    values=[1, 2, 3, 4, 5], value_rowids=[0, 0, 1, 1, 1]
)

# 2. Reversed keyword order (value_rowids, then values)
rt2 = RaggedTensor.from_value_rowids(
    value_rowids=[0, 0, 1, 1, 1], values=[1, 2, 3, 4, 5]
)

# 3. Mix of positional (values) and keyword (value_rowids)
rt3 = RaggedTensor.from_value_rowids([1, 2, 3, 4, 5], value_rowids=[0, 0, 1, 1, 1])

# 4. Explicit nrows as keyword
rt4 = RaggedTensor.from_value_rowids(
    values=[1, 2, 3, 4, 5], value_rowids=[0, 0, 1, 1, 1], nrows=3
)

test_keywords(rt1, rt2, rt3, rt4)
