# Test `ContainerElements` where the program deletes only by a name, an attribute, `del o.a`, or a
# string key, `del d["k"]`, in a function the call graph reaches and in one it does not: neither
# resizes a list, so a list keeps its length (wala/ML#1239).
import tensorflow as tf

x = tf.ones([2, 3])
y = tf.zeros([4], tf.int32)


def sink_list(c):
    assert len(c) == 2


class Holder:
    def __init__(self):
        self.a = 1


def delete_attribute(o):
    del o.a


def delete_key(d):
    del d["k"]


def never_called(o, d):
    del o.a
    del d["k"]


delete_attribute(Holder())
delete_key({"k": 1})
sink_list([x, y])
