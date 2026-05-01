# from https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/nn/sparse_softmax_cross_entropy_with_logits

import tensorflow as tf


def f(a):
    pass


logits = tf.constant(
    [[2.0, -5.0, 0.5, -0.1], [0.0, 0.0, 1.9, 1.4], [-100.0, 100.0, -100.0, -100.0]]
)
assert logits.shape == (3, 4)
assert logits.dtype == tf.float32

labels = tf.constant([0, 3, 1])
assert labels.shape == (3,)
assert labels.dtype == tf.int32

# The loss tensor returned by `tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)` has
# shape `labels.shape` (i.e., `(3,)`) and dtype `float32` (not `labels.dtype`), verified out-of-band
# with TF 2.x. Not asserted inline to avoid extracting an intermediate; see
# `TestTensorflow2Model.testSparseSoftmaxCrossEntropyWithLogits` Javadoc for the static-analysis
# anchor.
f(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits.numpy()))
