# Probe for wala/ML#745, class-method variant: two modules each define `Maker.make` with a
# different integer default, so any cross-module union of the intermediate default globals
# shows up as the sibling's shape in either sink.
import cls_a
import cls_b


def consume(t):
    pass


def consume2(t):
    pass


a = cls_a.Maker().make()
b = cls_b.Maker().make()
consume(a)
consume2(b)

assert a.shape == (4, 2)
assert b.shape == (5, 3)
