import tensorflow as tf

# Shape (2,)
a = tf.constant([1.0, 3.0])
print("Shape (2,): ", a)
assert a.shape == (2,)

# Shape (2, 1)
b = tf.constant([[1.0], [3.0]])
print("Shape (2, 1): ", b)
assert b.shape == (2, 1)

# Shape (1, 2)
c = tf.constant([[1.0, 3.0]])
print("Shape (1, 2): ", c)
assert c.shape == (1, 2)

# Shape (2, 3, 4): 2 blocks, each with 3 rows and 4 columns
d = tf.constant(
    [
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
        [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]],
    ]
)
print("Shape (2, 3, 4): ", d)
assert d.shape == (2, 3, 4)

# Shape (2, 3, 3): 2 blocks, each with 3 rows and 3 columns
e = tf.constant(
    [
        [[1, 2, 3], [5, 6, 7], [9, 10, 11]],
        [[13, 14, 15], [17, 18, 19], [21, 22, 23]],
    ]
)
print("Shape (2, 3, 3): ", e)
assert e.shape == (2, 3, 3)
