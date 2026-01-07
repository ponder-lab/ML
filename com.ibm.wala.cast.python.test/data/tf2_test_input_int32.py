import tensorflow as tf

def check_input(i1, i2):
    pass

# Test Input with explicit dtype=int32 and batch_size
# Expected: (32, 10) of int32
input1 = tf.keras.Input(shape=(10,), batch_size=32, dtype=tf.int32)
assert input1.dtype == tf.int32
assert input1.shape == (32, 10)

# Test Input with explicit dtype='int32' (string literal) and batch_size
# Expected: (8, 5, 5) of int32
input2 = tf.keras.Input(shape=(5, 5), batch_size=8, dtype='int32')
assert input2.dtype == tf.int32
assert input2.shape == (8, 5, 5)

check_input(input1, input2)
