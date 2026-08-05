import tensorflow as tf
from tensorflow.keras import layers


def consume_conv(t):
    pass


def consume_plain(t):
    pass


# A `Conv2D` layer call in the chain: the model's output shape is lost, so the consumer of the
# call result gets no rank. Minimal form of what `aymericdamien/TensorFlow-Examples`'s `ConvNet`
# does before handing `pred` to `accuracy` and `cross_entropy_loss`.
class ConvNet(tf.keras.Model):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = layers.Conv2D(32, kernel_size=5)
        self.flatten = layers.Flatten()
        self.out = layers.Dense(10)

    def call(self, x):
        return self.out(self.flatten(self.conv1(x)))


# Control: the same chain without the convolution keeps its shape all the way to the consumer.
class PlainNet(tf.keras.Model):
    def __init__(self):
        super(PlainNet, self).__init__()
        self.flatten = layers.Flatten()
        self.out = layers.Dense(10)

    def call(self, x):
        return self.out(self.flatten(x))


conv_net = ConvNet()
conv_pred = conv_net(tf.ones((128, 28, 28, 1)))
assert conv_pred.shape == (128, 10) and conv_pred.dtype == tf.float32
consume_conv(conv_pred)

plain_net = PlainNet()
plain_pred = plain_net(tf.ones((128, 28, 28, 1)))
assert plain_pred.shape == (128, 10) and plain_pred.dtype == tf.float32
consume_plain(plain_pred)
