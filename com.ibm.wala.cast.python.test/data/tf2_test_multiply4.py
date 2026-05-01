# From https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/math/multiply#for_example/

import tensorflow as tf


def f(a):
    pass


# Shape: (2, 3)
matrix = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
assert len(matrix) == 2 and len(matrix[0]) == 3  # Confirming shape (2, 3)
assert matrix[0][0].__class__ == float  # Confirming dtype float32

# Shape: (2, 1) -> Broadcasts columns to match matrix width (3)
col_vector = [[2.0], [3.0]]
assert len(col_vector) == 2 and len(col_vector[0]) == 1  # Confirming shape (2, 1)
assert col_vector[0][0].__class__ == float  # Confirming dtype float32

# 2. Column Vector Multiplication
# [1, 2, 3] * 2 = [2, 4, 6]
# [4, 5, 6] * 3 = [12, 15, 18]
result_col = tf.multiply(matrix, col_vector)
assert result_col.shape == (2, 3)
assert result_col.dtype == tf.float32

f(result_col)
