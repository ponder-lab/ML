import tensorflow as tf
import numpy as np


# Mirrors `MaskSparseCategoricalCrossentropy.__call__` from
# `kyzhouhzau/NLPGNN/nlpgnn/metrics/Losess.py`, a real-world NLP utility (a
# mask-weighted sparse-categorical-crossentropy loss), for tensor-type inference
# coverage.
class MaskSparseCategoricalCrossentropy:
    def __init__(self, from_logits=False, use_mask=False):
        self.from_logits = from_logits
        self.use_mask = use_mask

    def __call__(self, y_true, y_predict, input_mask=None):
        cross_entropy = tf.keras.losses.sparse_categorical_crossentropy(
            y_true, y_predict, self.from_logits
        )
        if self.use_mask:
            input_mask = tf.cast(input_mask, dtype=tf.float32)
            input_mask /= tf.reduce_mean(input_mask)
            cross_entropy *= input_mask
            # mask loss
            return tf.reduce_mean(cross_entropy)
        else:
            return tf.reduce_mean(cross_entropy)


loss = MaskSparseCategoricalCrossentropy(from_logits=True, use_mask=True)
y_true = tf.constant(np.ones((4,), dtype=np.int32))
y_predict = tf.constant(np.ones((4, 10), dtype=np.float32))
input_mask = tf.constant(np.ones((4,), dtype=np.float32))
result = loss(y_true, y_predict, input_mask)
assert y_true.shape == (4,) and y_true.dtype == tf.int32
assert y_predict.shape == (4, 10) and y_predict.dtype == tf.float32
assert input_mask.shape == (4,) and input_mask.dtype == tf.float32
assert result.shape == () and result.dtype == tf.float32
