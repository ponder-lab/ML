import tensorflow as tf


def consume_direct(pred):
    pass


def consume_two_context(pred):
    pass


def consume_conv(pred):
    pass


def consume_lstm(pred):
    pass


class Net(tf.keras.Model):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = tf.keras.layers.Dense(10)

    def call(self, x):
        return self.fc(x)


net = Net()

# Single calling context: the model-call result is consumed where it is produced.
pred = net(tf.ones((32, 784)))
assert pred.shape == (32, 10), pred.shape
consume_direct(pred)


# Two calling contexts differing only in batch size, which is the corpus shape: a train step and
# an evaluation step over the same model. The result's shape differs per context, so a correct
# answer is the union rather than either one alone.
def step(batch):
    out = net(batch)
    consume_two_context(out)
    return out


train_out = step(tf.ones((32, 784)))
assert train_out.shape == (32, 10), train_out.shape

test_out = step(tf.ones((10000, 784)))
assert test_out.shape == (10000, 10), test_out.shape


# A convolutional chain rather than a Dense one. The Dense case above resolves; this pins whether
# a Conv2D/pool/flatten chain's result carries its shape the same way.
class ConvNet(tf.keras.Model):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, 5, activation=tf.nn.relu)
        self.maxpool1 = tf.keras.layers.MaxPool2D(2, strides=2)
        self.flatten = tf.keras.layers.Flatten()
        self.out = tf.keras.layers.Dense(10)

    def call(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.flatten(x)
        return self.out(x)


conv_net = ConvNet()
conv_pred = conv_net(tf.ones((32, 28, 28, 1)))
assert conv_pred.shape == (32, 10), conv_pred.shape
consume_conv(conv_pred)


# A recurrent chain, the other family that appears in the corpus.
class LstmNet(tf.keras.Model):
    def __init__(self):
        super(LstmNet, self).__init__()
        self.lstm = tf.keras.layers.LSTM(32)
        self.out = tf.keras.layers.Dense(10)

    def call(self, x):
        return self.out(self.lstm(x))


lstm_net = LstmNet()
lstm_pred = lstm_net(tf.ones((32, 28, 28)))
assert lstm_pred.shape == (32, 10), lstm_pred.shape
consume_lstm(lstm_pred)


# The same convolutional chain with no pooling layer, which is wala/ML#820's original witness.
# Isolates whether the loss is the convolution or the pooling layer beside it.
class ConvNoPool(tf.keras.Model):
    def __init__(self):
        super(ConvNoPool, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, 5, activation=tf.nn.relu)
        self.flatten = tf.keras.layers.Flatten()
        self.out = tf.keras.layers.Dense(10)

    def call(self, x):
        return self.out(self.flatten(self.conv1(x)))


def consume_conv_nopool(pred):
    pass


nopool_net = ConvNoPool()
nopool_pred = nopool_net(tf.ones((32, 28, 28, 1)))
assert nopool_pred.shape == (32, 10), nopool_pred.shape
consume_conv_nopool(nopool_pred)
