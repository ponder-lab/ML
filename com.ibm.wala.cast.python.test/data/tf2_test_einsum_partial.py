# Test wala/ML#737: a partial einsum output composes from the resolved operand alone.
# `x` is the result of `tf.roll`, which the analysis does not model, so the first
# operand's shape is unresolved while `w` is statically known: the equation still
# proves the output rank and the `H` axis.
import tensorflow as tf


def consume(t):
    assert t.shape == (2, 4, 6)
    assert t.dtype == tf.float32


x = tf.roll(tf.ones((2, 4, 3, 5)), shift=1, axis=0)
w = tf.ones((3, 5, 6))
out = tf.einsum("BFND,NDH->BFH", x, w)
consume(out)
