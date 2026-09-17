# Three subclasses of a base imported from another module, at three nesting depths: top level,
# inside a function, inside a class. Each is driven with a distinct shape so the inherited
# method's parameter tells which of them resolved their base (wala/ML#944).
from zeta_base import Base


class Sub(Base):
    pass


def scale_inside(x):
    class Inner(Base):
        pass

    return Inner().scale(x)


class Outer:
    class Nested(Base):
        pass
