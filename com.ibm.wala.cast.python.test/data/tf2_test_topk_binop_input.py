# Faithful reduction of the corpus top_k row: the top_k INPUT is an elementwise binop of two resolved
# tensors (logits + labels * scalar), not a plain local. The binop result is overlay-resolved and its
# points-to set at the top_k synthetic node is empty, so TopK.composedShapes returns ⊤ at the
# inputPts guard before the sentinel logic runs. This reproduces the way column_indices fails in the
# corpus, which the plain-local fixture did not.
import tensorflow as tf


def consume_indices(v):
    pass


def consume_values(v):
    pass


logits = tf.constant([[1.0, 3.0, 2.0, 5.0, 4.0], [5.0, 4.0, 3.0, 2.0, 1.0]])  # (2, 5)
labels = tf.constant([[0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0]])  # (2, 5)

scored = logits + labels * 1000.0
k = tf.minimum(3, tf.shape(logits)[1])
values, indices = tf.nn.top_k(scored, k=k, sorted=False)
assert indices.shape == (2, 3)
consume_indices(indices)
consume_values(values)
