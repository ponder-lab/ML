import tensorflow as tf

# wala/ML#832: a `Sequential` whose LIST elements are USER Model subclasses. The blocks are the
# pyramid subject's residual block copied near-verbatim; the driver holds them in a constructor
# literal so the list reads whole, and what is exercised is the body fold: each block's `call` is
# never invoked, so its transform must come from walking the body itself. The shortcut attribute
# is path-insensitively BOTH construction branches (a projection stage and an identity lambda),
# and only the broadcast-coherent member pairs with the residual add.


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


stack = tf.keras.Sequential([Block(3, 6, strides=2), Block(6, 6)])
result = stack(tf.ones((1, 8, 8, 3)))
assert result.shape == (1, 4, 4, 6), result.shape
consume_stack(result)
