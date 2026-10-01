# Test https://github.com/wala/ML/issues/210.
import tensorflow as tf


def f(x):
    # Called as `pkg.sub.f(...)` after `import pkg`.
    return x + 1


def g(x):
    # Called as `g(...)` after `from pkg.sub import g`.
    return x + 1


def h(x):
    # Called as `pkg.sub.mod.h(...)` after `import pkg`.
    return x + 1


class Scale:
    def __call__(self, x):
        # Called as `pkg.sub.Scale()(...)` after `import pkg`.
        return x * 2


def f2(x):
    # Called as `sub.f2(...)` after `from pkg import sub`.
    return x + 1


def f3(x):
    # Called as `pkg.sub.f3(...)` after `import pkg.sub`.
    return x + 1


def f4(x):
    # Called as `m.f4(...)` after `import pkg.sub.mod as m`.
    return x + 1


def f5(x):
    # Called as `s.f5(...)` after `import pkg` and `s = pkg.sub`.
    return x + 1


def f6(x):
    # Called as `p.sub.f6(...)` after `import pkg as p`.
    return x + 1


def f7(x):
    # Called as `s.f7(...)` after `import pkg.sub as s`.
    return x + 1
