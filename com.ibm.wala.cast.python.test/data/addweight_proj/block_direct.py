import tensorflow as tf


def consume_plain(w):
    pass


class BlockPlain(tf.keras.layers.Layer):
    def build(self, input_shape):
        self.w = self.add_weight("w", shape=[4, 4], dtype=tf.float32)
        super(BlockPlain, self).build(input_shape)

    def call(self, x):
        consume_plain(self.w)
        return tf.matmul(x, self.w)
