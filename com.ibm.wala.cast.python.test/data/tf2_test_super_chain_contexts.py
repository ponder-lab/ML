# Test https://github.com/wala/ML/issues/995: layers built inside other layers' constructors, in
# loops and in lazy builds, each class a few `super().__init__(...)` levels deep and creating a
# weight per level. Each level's constructor must run on the one instance in its caller's context,
# so the weight creations are analysed once per instance and level, not multiplied by the depth of
# the super chain for every construction context.
import tensorflow as tf


def consume(w):
    assert w.shape == (2, 3)
    return w


class L0(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w0 = self.add_weight(shape=(2, 3), initializer="ones", name="w0")


class L1(L0):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w1 = self.add_weight(shape=(2, 3), initializer="ones", name="w1")


class Inner(L1):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.wb = self.add_weight(shape=(2, 3), initializer="ones", name="wb")
        super().build(input_shape)

    def call(self, inputs):
        return tf.matmul(inputs, self.w0 + self.w1 + self.wb)


class Block(L1):
    def __init__(self, n, **kwargs):
        super().__init__(**kwargs)
        self.inners = [Inner() for _ in range(n)]

    def build(self, input_shape):
        self.wc = self.add_weight(shape=(2, 3), initializer="ones", name="wc")
        for inner in self.inners:
            inner.build(input_shape)
        super().build(input_shape)

    def call(self, inputs):
        out = tf.matmul(inputs, self.wc)
        for inner in self.inners:
            out = out + inner(inputs)
        return out


class Net(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.b1 = Block(2)
        self.b2 = Block(3)
        self.blocks = [Block(1) for _ in range(2)]

    def call(self, inputs):
        out = self.b1(inputs) + self.b2(inputs)
        for b in self.blocks:
            out = out + b(inputs)
        consume(self.b1.w0)
        return out


net = Net()
net(tf.ones((4, 2)))
