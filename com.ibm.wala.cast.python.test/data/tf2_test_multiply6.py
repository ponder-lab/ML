# From https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/math/multiply#for_example/

import tensorflow as tf
from builtins import float


def f(a):
    pass


# Shape: (2, 3)
tensor_a = [[1, 2, 3], [4, 5, 6]]
assert len(tensor_a) == 2 and len(tensor_a[0]) == 3  # Confirm shape is (2, 3)
assert all(isinstance(x, int) for row in tensor_a for x in row)  # Confirm dtype is int

# Shape: (2,)
# Logic:
# 1. Align ranks -> (1, 2)
# 2. Compare right-to-left:
#    Dim 1: 3 vs 2 -> Mismatch! (Neither is 1)
tensor_b = [10, 20]
assert len(tensor_b) == 2  # Confirm shape is (2,)
assert all(isinstance(x, int) for x in tensor_b)  # Confirm dtype is int

try:
    result = tf.multiply(tensor_a, tensor_b)
    f(result)
except tf.errors.InvalidArgumentError as e:
    print("✅ Correctly caught expected error:")
    # We parse the string to show the relevant part of the error message
    print(f"Error Message: {e.message}")
