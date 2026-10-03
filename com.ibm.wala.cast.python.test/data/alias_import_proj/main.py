# Test wala/ML#1017: a submodule imported under an alias, `from pkg import mod as alias`, is the same
# module as `from pkg import mod`; a call through the alias reaches the submodule's function and class,
# and a class whose base is written through the submodule, `mod.Base` or `alias.Base`, has that base,
# so `super().__init__()` runs the base's initializer and the field it sets is read on the child.
from pkg import mod
from pkg import mod as alias


def consume_direct(t):
    assert t.shape == (2, 3)
    return t


def consume_alias(t):
    assert t.shape == (2, 3)
    return t


def consume_alias_class(t):
    assert t.shape == (5, 4)
    return t


def consume_base_field(t):
    assert t.shape == (2, 3)
    return t


def consume_alias_base_field(t):
    assert t.shape == (2, 3)
    return t


class Child(mod.Base):
    def __init__(self):
        super().__init__()


class AliasChild(alias.Base):
    def __init__(self):
        super().__init__()


consume_direct(mod.make())
consume_alias(alias.make())
consume_alias_class(alias.Maker(5).make())
consume_base_field(Child().t)
consume_alias_base_field(AliasChild().t)
