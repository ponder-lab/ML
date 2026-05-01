import tensorflow as tf


def check_case_1(rt):
    pass


# Case 1: Positional args, 1 ragged dim
flat_values = [3, 1, 4, 1, 5, 9]
rowids = [0, 0, 1, 3, 3, 3]
rt1 = tf.RaggedTensor.from_nested_value_rowids(flat_values, [rowids])
assert isinstance(rt1, tf.RaggedTensor)
assert rt1.shape.as_list() == [4, None]
assert rt1.dtype == tf.int32
check_case_1(rt1)


def check_case_2(rt):
    pass


# Case 2: Keyword args
rt2 = tf.RaggedTensor.from_nested_value_rowids(
    flat_values=flat_values, nested_value_rowids=[rowids]
)
assert isinstance(rt2, tf.RaggedTensor)
assert rt2.shape.as_list() == [4, None]
assert rt2.dtype == tf.int32
check_case_2(rt2)


def check_case_3(rt):
    pass


# Case 3: Mixed args
rt3 = tf.RaggedTensor.from_nested_value_rowids(
    flat_values, nested_value_rowids=[rowids]
)
assert isinstance(rt3, tf.RaggedTensor)
assert rt3.shape.as_list() == [4, None]
assert rt3.dtype == tf.int32
check_case_3(rt3)


def check_case_4(rt):
    pass


# Case 4: nested_nrows provided
rt4 = tf.RaggedTensor.from_nested_value_rowids(flat_values, [rowids], nested_nrows=[4])
assert isinstance(rt4, tf.RaggedTensor)
assert rt4.shape.as_list() == [4, None]
check_case_4(rt4)


def check_case_5(rt):
    pass


# Case 5: 2 ragged dimensions
flat_values_2 = [1, 2, 3, 4, 5, 6]
inner_rowids = [0, 0, 1, 2, 2, 3]
outer_rowids = [0, 0, 1, 1]
rt5 = tf.RaggedTensor.from_nested_value_rowids(
    flat_values_2, [outer_rowids, inner_rowids]
)
assert isinstance(rt5, tf.RaggedTensor)
assert rt5.shape.as_list() == [2, None, None]
assert rt5.dtype == tf.int32
check_case_5(rt5)


def check_case_6(rt):
    pass


# Case 6: Float values
flat_values_float = [1.0, 2.0]
rt6 = tf.RaggedTensor.from_nested_value_rowids(flat_values_float, [[0, 1]])
assert isinstance(rt6, tf.RaggedTensor)
assert rt6.shape.as_list() == [2, None]
assert rt6.dtype == tf.float32
check_case_6(rt6)


def check_case_7(rt):
    pass


# Case 7: Mixed args with nested_nrows
rt7 = tf.RaggedTensor.from_nested_value_rowids(flat_values, [rowids], nested_nrows=[4])
assert isinstance(rt7, tf.RaggedTensor)
assert rt7.shape.as_list() == [4, None]
check_case_7(rt7)
