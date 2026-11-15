# Test `ContainerElements` where a function the call graph does not reach resizes an attribute a dataclass's generated `__init__` stores (wala/ML#1255).
# The dataclass's generated initializer stores the global list under `self.items` (wala/ML#205).
import dataclasses
import threading

import tensorflow as tf

x = tf.ones([2, 3])
y = tf.zeros([4], tf.int32)


def sink_list(c):
    assert len(c) == 3, len(c)


@dataclasses.dataclass
class Holder:
    items: list

    def grow(self):
        self.items.append(1)


G = [x, y]
h = Holder(G)
t = threading.Thread(target=h.grow)
t.start()
t.join()
sink_list(G)
