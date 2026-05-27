# Reproducer for wala/ML#107 (method-inherited-from-parent missing from call graph)
# and wala/ML#118 (subclass.getSuperclass() returns Object instead of parent).


class D:
    def func(self, x):
        return x * x


class C(D):
    pass


c = C()
a = c.func(5)
assert a == 25
