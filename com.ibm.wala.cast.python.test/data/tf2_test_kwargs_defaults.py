import tensorflow as tf


def consume_first(a):
    pass


def consume_middle(b):
    pass


def consume_last(c):
    pass


# Python's positional defaults apply to the last parameters of the plain positional list, but a
# trailing `**kwargs` is a formal too. Anything locating the defaulted range by counting back from
# the end of the combined formal list binds each default to the parameter after the one it belongs
# to, and leaves the first defaulted parameter with nothing (wala/ML#843). Each default here is a
# DIFFERENT shape, so a shift by one slot is visible rather than masked by a shared value.
class Defaults:
    def __init__(
        self,
        required,
        first=(2, 3),
        middle=(4, 5, 6),
        last=(7,),
        **kwargs,
    ):
        self.required = required
        self.first = first
        self.middle = middle
        self.last = last

    def emit(self):
        consume_first(tf.zeros(self.first))
        consume_middle(tf.zeros(self.middle))
        consume_last(tf.zeros(self.last))


d = Defaults("r")
assert d.first == (2, 3), d.first
assert d.middle == (4, 5, 6), d.middle
assert d.last == (7,), d.last
d.emit()
