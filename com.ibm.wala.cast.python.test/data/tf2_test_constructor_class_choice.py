# Test: one call site that constructs either of two classes, through a variable bound to one or the
# other, runs each class's own `__init__` and binds each class's own methods on that class's
# instances only.
import tensorflow as tf

SEEN = []


def consume_a(t):
    SEEN.append(("a", t.shape.as_list()))
    return t


def consume_b(t):
    SEEN.append(("b", t.shape.as_list()))
    return t


class A:

    def __init__(self):
        self.t = tf.ones([2])

    def get(self):
        return consume_a(self.t)


class B:

    def __init__(self):
        self.t = tf.ones([3, 3])

    def get(self):
        return consume_b(self.t)


def make(flag):
    cls = A if flag else B
    return cls()


for flag in [True, False]:
    make(flag).get()

assert SEEN == [("a", [2]), ("b", [3, 3])], SEEN
