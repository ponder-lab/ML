# Witness for wala/ML#877: a shape-composing op (matmul) drops an operand's rank
# when the operand is typed only in dataflow state that PTS-based resolution
# cannot see, here an operand bound by tuple-unpacking. The operand's own type
# survives (a direct consumer reads it); only the matmul-mediated read floors to
# unknown rank. Fixed by a MatMul type feed. Without the fix the parameter is
# unknown-rank float32; with it, (?, 10) float32.
import tensorflow as tf
import numpy as np


def consume(x):
    pass


W = tf.ones([4, 10])
t = np.array(np.load("a.npy"), np.float32).reshape([-1, 4])
other, t = 0.0, t
consume(tf.matmul(t, W))
