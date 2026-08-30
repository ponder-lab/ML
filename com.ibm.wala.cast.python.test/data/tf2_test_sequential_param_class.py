import tensorflow as tf

# wala/ML#832: the residual-block stack of tf2_test_sequential_user_blocks.py, with ONE thing
# varied — the block CLASS reaches the constructing helper as a PARAMETER, the pyramid subject's
# own construction shape (`_make_layer(block, ...)`). The runtime is identical to the sibling
# fixture; what is measured is whether the body fold's class-identity derivation survives the
# parameter hop.


def consume_stack(x):
    assert isinstance(x, tf.Tensor)
    assert x.shape == (1, 4, 4, 6), x.shape
    assert x.dtype == tf.float32


class Block(tf.keras.Model):
    expansion = 1

    def __init__(self, in_channels, out_channels, strides=1):
        super(Block, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            out_channels, kernel_size=3, strides=strides, padding="same", use_bias=False
        )
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(
            out_channels, kernel_size=3, strides=1, padding="same", use_bias=False
        )
        self.bn2 = tf.keras.layers.BatchNormalization()

        if strides != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = tf.keras.Sequential(
                [
                    tf.keras.layers.Conv2D(
                        self.expansion * out_channels,
                        kernel_size=1,
                        strides=strides,
                        use_bias=False,
                    ),
                    tf.keras.layers.BatchNormalization(),
                ]
            )
        else:
            self.shortcut = lambda x, _: x

    def call(self, x, training=False):
        out = tf.nn.relu(self.bn1(self.conv1(x), training=training))
        out = self.bn2(self.conv2(out), training=training)
        out += self.shortcut(x, training)
        return tf.nn.relu(out)


def make_pair(block):
    return tf.keras.Sequential([block(3, 6, 2), block(6, 6)])


stack = make_pair(Block)
result = stack(tf.ones((1, 8, 8, 3)))
assert result.shape == (1, 4, 4, 6), result.shape
consume_stack(result)


def consume_stage_one(x):
    assert isinstance(x, tf.Tensor)
    assert x.shape == (1, 8, 8, 64), x.shape
    assert x.dtype == tf.float32


# The pyramid's first stage parameterization: in_channels equals out_channels and the stride is
# one, so the identity branch is the live shortcut on both blocks.
stage_one = (
    make_pair_same(Block)
    if False
    else tf.keras.Sequential([Block(64, 64, 1), Block(64, 64)])
)
result_one = stage_one(tf.ones((1, 8, 8, 64)))
assert result_one.shape == (1, 8, 8, 64), result_one.shape
consume_stage_one(result_one)
