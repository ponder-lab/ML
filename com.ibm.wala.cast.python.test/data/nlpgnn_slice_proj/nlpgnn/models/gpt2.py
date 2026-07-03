# Miniature of NLPGNN's `nlpgnn/models/gpt2.py` (wala/ML#678): the inner model both sibling
# scripts' `GenGPT2` wraps.
import tensorflow as tf


class GPT2(tf.keras.Model):
    def __init__(self, param, **kwargs):
        super(GPT2, self).__init__(**kwargs)
        self.param = param
        self.dense = tf.keras.layers.Dense(8)

    def call(self, inputs, past=None, is_training=True):
        return self.dense(inputs)
