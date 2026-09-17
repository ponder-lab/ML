# The other direction, deferred into a function body, because a module-level import of a here
# would leave one module partially initialised and Python would refuse to run the program. This is
# the only shape a base-dependency cycle can take and still execute (wala/ML#944).


class B:
    def scale(self, x):
        return x * 2.0


def make():
    from a import A

    class C(A):
        pass

    return C
