# A rank guard in a loop whose base is the loop's own φ: the guard governs an arm of a φ that flows
# back into its base, so reading the base's creators passes through an arm the guard decides.
import tensorflow as tf


def consume(x):
    pass


x = tf.ones([2, 2])
for _ in range(3):
    if x.shape.rank == 2:
        x = tf.reshape(x, [4])
assert x.shape == (4,)
consume(x)
