# A function that only a call in an arm a constant guard never takes reaches does not run in that
# context, so neither it nor a function it calls receives a value from there.
import tensorflow as tf


def through_helper(x):
    assert x.shape == (2, 3) and x.dtype == tf.float32


def helper(value):
    through_helper(tf.abs(value))


def through_chain(x):
    assert x.shape == (2, 3) and x.dtype == tf.float32


def chain_inner(value):
    through_chain(tf.abs(value))


def chain_outer(value):
    chain_inner(tf.negative(value))


def through_method(x):
    assert x.shape == (2, 3) and x.dtype == tf.float32


class Targets:
    def assign(self, value):
        through_method(tf.abs(value))


def through_build(x):
    assert x.shape == (2, 3) and x.dtype == tf.float32


def into_build(x):
    assert x.shape == (2, 3) and x.dtype == tf.float32


class Layers:
    def build(self, value):
        through_build(tf.abs(value))
        into_build(value)


def through_recursion(x):
    assert x.shape == (2, 3) and x.dtype == tf.float32


def ping(value, depth):
    through_recursion(tf.abs(value))
    if depth > 0:
        pong(value, depth - 1)


def pong(value, depth):
    ping(value, depth)


class Model:
    # Each function and method the arm calls has a node for each call of `run`, but for the method
    # named `build`, which has one node for each instance, so the call that passes `True` types it.
    def __init__(self):
        self.targets = Targets()
        self.layers = Layers()

    def run(self, values, flag):
        # The arm unpacks a third member, which the call that passes `False` does not hold.
        if flag:
            a, b, c = values
            helper(c)
            chain_outer(c)
            self.targets.assign(c)
            self.layers.build(c)
            ping(c, 2)
        self.done = True


x = tf.ones((4,))
y = tf.ones((5,))
z = tf.ones((2, 3))

model = Model()
model.run((x, y), False)
model.run((x, y, z), True)


def through_model_call(x):
    assert x.shape == (2, 3) and x.dtype == tf.float32


class Builder:
    def build_targets(self, batch):
        through_model_call(tf.abs(batch[0]))


class Detector(tf.keras.Model):
    # A model called for inference with two inputs and for training with three. The method of the
    # object the model holds has a node for each call of the model, and reads the first member of
    # the batch it is passed.
    def __init__(self):
        super(Detector, self).__init__()
        self.builder = Builder()

    def __call__(self, inputs, training=True):
        if training:
            a, b, c = inputs
        else:
            a, b = inputs
        if training:
            self.builder.build_targets(c)
        return a


detector = Detector()
detector((x, y), training=False)
detector((x, y, tf.ones((1, 2, 3))), training=True)


def never(x):
    return x


def never_helper(value):
    never(tf.abs(value))


def scaled(value, scale):
    assert value.shape == (2, 3) and scale is not None


def none_helper(value):
    scaled(tf.abs(value), None)


class Skipped:
    # No call passes `True`, so the functions the arm calls run in no context.
    def run(self, values, flag):
        if flag:
            a, b, c = values
            never_helper(c)
            none_helper(c)
        self.done = True


skipped = Skipped()
skipped.run((x, y), False)
scaled(z, z)


def through_shared(x):
    assert x.shape == (2, 3) and x.dtype == tf.float32


class Parts:
    def build(self, value):
        through_shared(tf.abs(value))


class Shared:
    # The call in the arm and the call below it reach the one node that a method named `build` has
    # for its instance, which the second call runs.
    def __init__(self):
        self.parts = Parts()

    def run(self, values, flag):
        a, b, c = values
        if flag:
            self.parts.build(c)
        self.parts.build(c)


shared = Shared()
shared.run((x, y, z), False)


def through_subscript(x):
    assert x.shape == (2, 3) and x.dtype == tf.float32


def unused_helper(value):
    return value


def in_subscript_method(x):
    assert x.shape == (2, 3) and x.dtype == tf.float32


class Boxes:
    def __getitem__(self, index):
        value = tf.ones((2, 3))
        in_subscript_method(value)
        return value


class Indexed:
    # The subscript calls `__getitem__` beside an arm that never runs.
    def run(self, boxes, flag):
        if flag:
            unused_helper(boxes)
        through_subscript(boxes[0])


indexed = Indexed()
indexed.run(Boxes(), False)
