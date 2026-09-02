# Witness for wala/ML#868 layer one: a decorator applied in CALL form carries its
# argument expressions in the CAst, and the loader mines the name-shaped ones
# (identifiers, dotted attributes, and the None constant) into a parallel channel
# beside the name-only annotations. The decoration itself is (still) never applied
# in IR, so the runner below exists only to keep the file runtime-faithful.
class A:
    pass


class B:
    pass


def params(*classes):
    def wrap(f):
        def run(receiver):
            for c in classes:
                if c is not None:
                    f(receiver, c)

        return run

    return wrap


def pick():
    return B


def identity(f):
    return f


class T:
    @params(A, B, None)
    def m(self, cls):
        return cls()

    # A call-expression argument has no name to mine; it must surface as the
    # explicit unmineable marker rather than silently vanishing, so a consumer
    # sees the argument COUNT the program has.
    @params(pick())
    def n(self, cls):
        return cls()

    # A bare (call-less) decorator: the front end normalizes it to a
    # zero-argument call in the CAst, so it appears with an empty argument list,
    # and its name-only annotation channel is unchanged.
    @identity
    def p(self):
        return 2


t = T()
t.m()
t.n()
assert t.p() == 2
