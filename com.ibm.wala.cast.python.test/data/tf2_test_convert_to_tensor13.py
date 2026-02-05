import tensorflow as tf


def test(t1, t2, t3):
    pass


# Mix of positional (value) and keyword (dtype)
t1 = tf.convert_to_tensor([1, 2, 3], dtype=tf.float32)
assert t1.shape == (3,)
assert t1.dtype == tf.float32

# Keyword (value) and positional (dtype - not possible to skip first pos arg, so this case is just keywords)
t2 = tf.convert_to_tensor(value=[[1, 2], [3, 4]], dtype=tf.int32)
assert t2.shape == (2, 2)
assert t2.dtype == tf.int32

# Just keyword value
t3 = tf.convert_to_tensor(value=[1.0, 2.0])
assert t3.shape == (2,)
assert t3.dtype == tf.float32

test(t1, t2, t3)
