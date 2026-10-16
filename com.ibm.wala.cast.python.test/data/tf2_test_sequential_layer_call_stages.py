import tensorflow as tf

# wala/ML#832: a generator whose `Sequential` is built from a list grown by `+=` in nested loops,
# holding two user layer classes the model alone calls. Each class is instantiated once per
# stage, and each stage feeds it a different shape, so its `call` sees every shape below at
# runtime.

TRANSPOSE_INPUTS = {(2, 10, 64), (2, 20, 32)}
STACK_INPUTS = {(2, 20, 32), (2, 40, 16)}


class TFReflectionPad1d(tf.keras.layers.Layer):

    def __init__(self, padding_size, **kwargs):
        super().__init__(**kwargs)
        self.padding_size = padding_size

    def call(self, x):
        return tf.pad(
            x, [[0, 0], [self.padding_size, self.padding_size], [0, 0]], "REFLECT"
        )


class TFConvTranspose1d(tf.keras.layers.Layer):

    def __init__(self, filters, kernel_size, strides, **kwargs):
        super().__init__(**kwargs)
        self.conv1d_transpose = tf.keras.layers.Conv2DTranspose(
            filters=filters,
            kernel_size=(kernel_size, 1),
            strides=(strides, 1),
            padding="same",
        )

    def call(self, x):
        assert tuple(x.shape) in TRANSPOSE_INPUTS, x.shape
        assert x.dtype == tf.float32
        x = tf.expand_dims(x, 2)
        x = self.conv1d_transpose(x)
        x = tf.squeeze(x, 2)
        return x


class TFResidualStack(tf.keras.layers.Layer):

    def __init__(self, kernel_size, filters, dilation_rate, **kwargs):
        super().__init__(**kwargs)
        self.blocks = [
            tf.keras.layers.LeakyReLU(alpha=0.2),
            TFReflectionPad1d((kernel_size - 1) // 2 * dilation_rate),
            tf.keras.layers.Conv1D(
                filters=filters, kernel_size=kernel_size, dilation_rate=dilation_rate
            ),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.Conv1D(filters=filters, kernel_size=1),
        ]
        self.shortcut = tf.keras.layers.Conv1D(filters=filters, kernel_size=1)

    def call(self, x):
        assert tuple(x.shape) in STACK_INPUTS, x.shape
        assert x.dtype == tf.float32
        _x = tf.identity(x)
        for layer in self.blocks:
            _x = layer(_x)
        shortcut = self.shortcut(x)
        return shortcut + _x


class TFMelGANGenerator(tf.keras.Model):

    def __init__(self, filters=64, upsample_scales=(2, 2), stacks=2, **kwargs):
        super().__init__(**kwargs)
        layers = []
        layers += [
            TFReflectionPad1d(3),
            tf.keras.layers.Conv1D(filters=filters, kernel_size=7),
        ]
        for i, upsample_scale in enumerate(upsample_scales):
            layers += [
                tf.keras.layers.LeakyReLU(alpha=0.2),
                TFConvTranspose1d(
                    filters=filters // (2 ** (i + 1)),
                    kernel_size=upsample_scale * 2,
                    strides=upsample_scale,
                ),
            ]
            for j in range(stacks):
                layers += [
                    TFResidualStack(
                        kernel_size=3,
                        filters=filters // (2 ** (i + 1)),
                        dilation_rate=3**j,
                    )
                ]
        layers += [
            tf.keras.layers.LeakyReLU(alpha=0.2),
            TFReflectionPad1d(3),
            tf.keras.layers.Conv1D(filters=1, kernel_size=7),
        ]
        self.melgan = tf.keras.models.Sequential(layers)

    def call(self, mels):
        return self.melgan(mels)


generator = TFMelGANGenerator()
audio = generator(tf.random.normal([2, 10, 80]))
assert audio.shape == (2, 40, 1), audio.shape
