# from https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/nn/sparse_softmax_cross_entropy_with_logits

import tensorflow as tf


def f(a):
    # Runtime anchor for the static-analysis expectation in
    # `TestTensorflow2Model.testSparseSoftmaxCrossEntropyWithLogits`: the loss is a fresh tensor of
    # shape `labels.shape` and dtype `float32` — NOT a view of `labels` (the pre-wala/ML#412
    # modeling reported `(3,) int32` via a `<return value="labels"/>` pass-through artefact).
    assert a.shape == (3,)
    assert a.dtype == tf.float32


logits = tf.constant(
    [[2.0, -5.0, 0.5, -0.1], [0.0, 0.0, 1.9, 1.4], [-100.0, 100.0, -100.0, -100.0]]
)
assert logits.shape == (3, 4)
assert logits.dtype == tf.float32

labels = tf.constant([0, 3, 1])
assert labels.shape == (3,)
assert labels.dtype == tf.int32

f(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits.numpy()))
