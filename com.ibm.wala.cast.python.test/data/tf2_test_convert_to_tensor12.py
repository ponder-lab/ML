import tensorflow as tf


def f(a):
    pass


# A 2D list (Matrix)
matrix_list = [[1, 2, 3], [4, 5, 6]]

assert isinstance(matrix_list, list)
assert len(matrix_list) == 2
assert all(isinstance(row, list) for row in matrix_list)
assert all(isinstance(x, int) for row in matrix_list for x in row)
assert len(matrix_list[0]) == 3
assert len(matrix_list[1]) == 3

# Convert the 2D list to a TensorFlow Tensor
matrix_tensor = tf.convert_to_tensor(matrix_list)

# Output: shape=(2, 3), dtype=int32

assert isinstance(matrix_tensor, tf.Tensor)
assert matrix_tensor.dtype == tf.int32
assert matrix_tensor.shape == (2, 3)

f(matrix_tensor)
