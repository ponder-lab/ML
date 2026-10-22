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


# The initializers alone: both write `t`, and differently named methods read it, so only an
# initializer of the other class can put the other tensor in a sink.
def consume_c(t):
    SEEN.append(("c", t.shape.as_list()))
    return t


def consume_d(t):
    SEEN.append(("d", t.shape.as_list()))
    return t


class C:

    def __init__(self):
        self.t = tf.ones([2])

    def get_c(self):
        return consume_c(self.t)


class D:

    def __init__(self):
        self.t = tf.ones([3, 3])

    def get_d(self):
        return consume_d(self.t)


def make_init_only(flag):
    cls = C if flag else D
    return cls()


make_init_only(True).get_c()
make_init_only(False).get_d()


# Class-body attributes alone: no initializer, and differently named methods read them.
def consume_e(t):
    SEEN.append(("e", t.shape.as_list()))
    return t


def consume_f(t):
    SEEN.append(("f", t.shape.as_list()))
    return t


class E:
    k = tf.ones([2])

    def get_e(self):
        return consume_e(self.k)


class F:
    k = tf.ones([3, 3])

    def get_f(self):
        return consume_f(self.k)


def make_attribute_only(flag):
    cls = E if flag else F
    return cls()


make_attribute_only(True).get_e()
make_attribute_only(False).get_f()

assert SEEN[2:] == [("c", [2]), ("d", [3, 3]), ("e", [2]), ("f", [3, 3])], SEEN
