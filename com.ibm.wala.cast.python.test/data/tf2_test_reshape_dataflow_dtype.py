import tensorflow as tf


def consume(t):
    pass


class Block(tf.keras.layers.Layer):
    def build(self, input_shape):
        self.w = self.add_weight("w", shape=[4, 4], dtype=tf.float32)
        self.b = self.add_weight("b", shape=[4], dtype=tf.float32)
        super(Block, self).build(input_shape)

    def call(self, x):
        flat = tf.reshape(x, [-1, 4])
        consume(flat)
        y = tf.matmul(flat, self.w) + self.b
        return tf.reshape(y, [2, 3, 4])


class Stack(tf.keras.Model):
    def __init__(self):
        super(Stack, self).__init__()
        self.first = Block()
        self.second = Block()

    def call(self, x):
        return self.second(self.first(x))


model = Stack()
out = model(tf.ones((2, 3, 4)))
assert out.shape == (2, 3, 4) and out.dtype == tf.float32
