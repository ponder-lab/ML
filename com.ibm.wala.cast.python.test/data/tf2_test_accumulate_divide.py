import tensorflow as tf

# wala/ML#1094: a tensor accumulated over a list of layers in a loop (`xs = None`, then
# `xs += block(x)`) and divided afterwards. The loop variable and the `+=` read each other, so the
# first read of either is made while the other is still unresolved. Each spelling has its own
# classes, so each consumer's parameter is fed by its spelling alone.


class Res(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv = tf.keras.layers.Conv1D(8, 3, padding="same")

    def call(self, x):
        return self.conv(x) + x


class LoopLen(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.blocks = [Res(), Res()]

    def call(self, x):
        xs = None
        for block in self.blocks:
            if xs is None:
                xs = block(x)
            else:
                xs += block(x)
        return xs / len(self.blocks)


class LoopTwo(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.blocks = [Res(), Res()]

    def call(self, x):
        xs = None
        for block in self.blocks:
            if xs is None:
                xs = block(x)
            else:
                xs += block(x)
        return xs / 2


class NoLoopLen(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.blocks = [Res(), Res()]

    def call(self, x):
        xs = self.blocks[0](x)
        xs += self.blocks[1](x)
        return xs / len(self.blocks)


class ConsumeLoopLen(tf.keras.layers.Layer):

    def call(self, x):
        assert x.shape == (2, 5, 8), x.shape
        assert x.dtype == tf.float32
        return x


class ConsumeLoopTwo(tf.keras.layers.Layer):

    def call(self, x):
        assert x.shape == (2, 5, 8), x.shape
        assert x.dtype == tf.float32
        return x


class ConsumeNoLoopLen(tf.keras.layers.Layer):

    def call(self, x):
        assert x.shape == (2, 5, 8), x.shape
        assert x.dtype == tf.float32
        return x


ConsumeLoopLen()(LoopLen()(tf.random.normal([2, 5, 8])))
ConsumeLoopTwo()(LoopTwo()(tf.random.normal([2, 5, 8])))
ConsumeNoLoopLen()(NoLoopLen()(tf.random.normal([2, 5, 8])))
