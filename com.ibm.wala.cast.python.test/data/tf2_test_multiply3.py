# From https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/math/multiply#for_example/

import tensorflow as tf


def f(a):
    pass


# Shape: (2, 3)
matrix = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
assert len(matrix) == 2 and len(matrix[0]) == 3  # Confirming shape (2, 3)
assert matrix[0][0].__class__ == float  # Confirming dtype float32

# Shape: (1,) -> Broadcasts to (2, 3)
scalar = [10.0]
assert len(scalar) == 1  # Confirming shape (1,)
assert scalar[0].__class__ == float  # Confirming dtype float32

# 1. Scalar Multiplication
result_scalar = tf.multiply(matrix, scalar)
assert result_scalar.shape == (2, 3)
assert result_scalar.dtype == tf.float32

f(result_scalar)
