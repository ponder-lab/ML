import tensorflow as tf
from block_wild import BlockWild
from block_direct import BlockPlain
from block_shadow import BlockShadow


class Stack(tf.keras.Model):
    def __init__(self):
        super(Stack, self).__init__()
        self.plain = BlockPlain()
        self.wild = BlockWild()

    def call(self, x):
        return self.wild(self.plain(x))


model = Stack()
out = model(tf.ones((2, 4)))
assert out.shape == (2, 4) and out.dtype == tf.float32

shadow = BlockShadow().run()
assert shadow.shape == (3, 3) and shadow.dtype == tf.float32
