import tensorflow as tf
from tensorflow.python import keras


class TransformerLoss(keras.losses.SparseCategoricalCrossentropy):
    def __init__(self, from_logits=False, reduction="none", debug=False, **kwargs):
        super(TransformerLoss, self).__init__(from_logits, reduction, **kwargs)
        self.debug = debug

    def call(self, y_true, y_pred):
        assert y_true.shape == (4,) and y_true.dtype == tf.int64
        assert y_pred.shape == (4, 3) and y_pred.dtype == tf.float32
        y_true = tf.cast(y_true, tf.int32)
        return super(TransformerLoss, self).call(y_true, y_pred)
