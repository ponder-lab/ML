# Test https://github.com/wala/ML/issues/1249: a layer called without `training`, or with `None`,
# inside a layer call that has a `training` value takes that value from Keras's call context, and
# only without one its own default.
import tensorflow as tf

SEEN = []


class Nested(tf.keras.layers.Layer):

    def call(self, x, training=None):
        SEEN.append(("Nested", training))
        return x


class Outer(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.inner = Nested()

    def call(self, x, training=None):
        return self.inner(x)


class Explicit(tf.keras.layers.Layer):

    def call(self, x, training=None):
        SEEN.append(("Explicit", training))
        return x


class OverridesContext(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.inner = Explicit()

    def call(self, x, training=None):
        return self.inner(x, training=False)


class Helped(tf.keras.layers.Layer):

    def call(self, x, training=None):
        SEEN.append(("Helped", training))
        return x


def helper(layer, x):
    return layer(x)


class ThroughHelper(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.inner = Helped()

    def call(self, x, training=None):
        return helper(self.inner, x)


class Passed(tf.keras.layers.Layer):

    def call(self, x, training=None):
        SEEN.append(("Passed", training))
        return x


class NoTraining(tf.keras.layers.Layer):

    def __init__(self):
        super().__init__()
        self.inner = Passed()

    def call(self, x):
        return self.inner(x)


class ThroughNoTraining(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.middle = NoTraining()

    def call(self, x, training=None):
        return self.middle(x)


class Direct(tf.keras.layers.Layer):

    def call(self, x, training=None):
        SEEN.append(("Direct", training))
        return x


class CallsDirectly(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.inner = Direct()

    def call(self, x, training=None):
        return self.inner.call(x)


class Defaulted(tf.keras.layers.Layer):

    def call(self, x, training=False):
        SEEN.append(("Defaulted", training))
        return x


class PassesNone(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.inner = Defaulted()

    def call(self, x, training=None):
        return self.inner(x, training=None)


xs = tf.ones([2, 3])
Outer()(xs, training=True)
OverridesContext()(xs, training=True)
ThroughHelper()(xs, training=True)
ThroughNoTraining()(xs, training=True)
CallsDirectly()(xs, training=True)
PassesNone()(xs)

assert set(SEEN) == {
    ("Nested", True),
    ("Explicit", False),
    ("Helped", True),
    ("Passed", True),
    ("Direct", None),
    ("Defaulted", False),
}, SEEN
