# Test constant guards whose arm never runs: one an argument decides (wala/ML#1187) and one a field decides (wala/ML#1256).
# Each `Conv`-shaped class's dead arm reaches the code after it by a different route, or by none.
import tensorflow as tf


def consume(y_pred):
    assert not isinstance(y_pred, tuple)
    assert y_pred.shape == (2, 4)
    assert y_pred.dtype == tf.float32


class Net(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.fc = tf.keras.layers.Dense(4)

    def call(self, inputs, training=None, eval=None):
        fc = self.fc(inputs)
        if training:
            return fc
        elif eval:
            return fc, fc
        else:
            return tf.nn.softmax(fc)

    def custom_train_step(self, x):
        predictions = self.call(x, training=True)
        consume(predictions)


net = Net()
net.custom_train_step(tf.ones([2, 3]))


def add_loop(edges, n):
    rows = edges[:, 0]
    return tf.concat([edges, tf.stack([rows, rows], 1)], 0)


def add_loop_nosub(edges, n):
    return tf.concat(edges, 0)


def flag_full(f):
    assert f is True


def flag_noassign(f):
    assert f is True


def message_full(state, index):
    assert index == 0


def message_noassign(state, index):
    assert index == 0


def message_nosub(state, index):
    assert index == 0


def message_none(state, index):
    assert index == 0


class Full:
    def __init__(self, concat=True):
        self.concat = concat

    def call(self, x, adj):
        flag_full(self.concat)
        if not self.concat:
            adj = add_loop(adj, len(x))
        for index, a in enumerate(adj):
            message_full(a, index)


class NoAssign:
    def __init__(self, concat=True):
        self.concat = concat

    def call(self, x, adj):
        flag_noassign(self.concat)
        if not self.concat:
            add_loop(adj, len(x))
        for index, a in enumerate(adj):
            message_noassign(a, index)


class NoSub:
    def __init__(self, concat=True):
        self.concat = concat

    def call(self, x, adj):
        if not self.concat:
            adj = add_loop_nosub(adj, len(x))
        for index, a in enumerate(adj):
            message_nosub(a, index)


def message_bind(state, index):
    assert index == 0


class BindOnly:
    def __init__(self, concat=True):
        self.concat = concat

    def call(self, x, adj):
        if not self.concat:
            add_loop_nosub(adj, len(x))
        for index, a in enumerate(adj):
            message_bind(a, index)


class NoArm:
    def __init__(self, concat=True):
        self.concat = concat

    def call(self, x, adj):
        for index, a in enumerate(adj):
            message_none(a, index)


x = tf.ones([3, 4])
Full().call(x, [tf.ones([3, 2], tf.int32)])
NoAssign().call(x, [tf.ones([3, 2], tf.int32)])
NoSub().call(x, [tf.ones([3, 2], tf.int32)])
BindOnly().call(x, [tf.ones([3, 2], tf.int32)])
NoArm().call(x, [tf.ones([3, 2], tf.int32)])
