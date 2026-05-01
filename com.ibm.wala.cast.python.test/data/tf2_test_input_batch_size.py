import tensorflow as tf


def check_input(i1, i2, i3):
    pass


# Test Input with explicit batch_size
# Expected: (16, 32)
input1 = tf.keras.Input(shape=(32,), batch_size=16)
assert input1.shape == (16, 32)
assert input1.dtype == tf.float32

# Test Input with explicit batch_size and different dims
# Expected: (5, 10, 10)
input2 = tf.keras.Input(shape=(10, 10), batch_size=5)
assert input2.shape == (5, 10, 10)
assert input2.dtype == tf.float32

# Test with None batch_size (explicitly passed)
# Expected: (None, 5)
input3 = tf.keras.Input(shape=(5,), batch_size=None)
assert input3.shape == (None, 5)
assert input3.dtype == tf.float32

check_input(input1, input2, input3)
