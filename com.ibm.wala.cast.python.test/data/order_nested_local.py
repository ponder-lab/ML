import tensorflow as tf


# Same-module twin of the order fixture's function-nested arm (wala/ML#944): the base is local, so
# module order cannot matter, and the arm isolates whether the engine dispatches an inherited method
# on a class defined inside a function at all.
class Base:
    def scale(self, x):
        return x * 2.0


def scale_inside(x):
    class Inner(Base):
        pass

    return Inner().scale(x)


b = scale_inside(tf.ones((4,)))
assert b.shape == (4,)
