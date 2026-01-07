import tensorflow as tf


def check_input(i1, i2, i3):
    pass


# 1. Positional arguments only: shape, batch_size, name, dtype
# Input((10,), 32, "my_input", tf.int32)
# shape=(10,), batch_size=32, dtype=int32
input1 = tf.keras.Input((10,), 32, "input1", tf.int32)
assert input1.shape == (32, 10)
assert input1.dtype == tf.int32

# 2. Positional and Keyword arguments: shape as positional, batch_size as keyword
# Input((5, 5), batch_size=16)
# shape=(5, 5), batch_size=16, dtype=float32 (default)
input2 = tf.keras.Input((5, 5), batch_size=16)
assert input2.shape == (16, 5, 5)
assert input2.dtype == tf.float32

# 3. Positional and Keyword: shape as positional, dtype as keyword
# Input((20,), dtype="int32")
# shape=(20,), batch_size=None (default), dtype=int32
input3 = tf.keras.Input((20,), dtype="int32")
assert input3.shape == (None, 20)
assert input3.dtype == tf.int32

check_input(input1, input2, input3)
