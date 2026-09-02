# Witness fixture for wala/ML#868 layer two: a pytest-shaped test class whose
# methods' class parameters are supplied ONLY by decorator arguments. The
# runtime driver at the bottom mirrors what the parameterized runner does, so
# the file runs standalone; the analysis-side supply comes from the entrypoint
# binding, since decoration is never applied in IR.
import parameterized
import parameterized_classes
from parameterized_classes import A


def notparameters(*args):
    # Same shape and runtime behavior as `parameterized.parameters`, but not a
    # recognized name: the binding is name-gated, and a decorator with different
    # runtime semantics must not inject values.
    def wrap(f):
        def run(receiver):
            for a in args:
                f(receiver, a)

        return run

    return wrap


class TestPick:
    @parameterized.parameters(parameterized_classes.A, parameterized_classes.B, None)
    def test_pick(self, cls):
        # The None arm mirrors the subject's guard; its test case contributes no
        # analysis evidence until a null-constant binding lands.
        if cls is not None:
            x = cls()
            assert x.tag() in (1, 2)

    @notparameters(parameterized_classes.A)
    def test_unrecognized(self, cls):
        if cls is not None:
            cls()

    @parameterized.parameters(A)
    def test_bare(self, cls):
        # A from-imported bare name: the class is defined in the other module,
        # so resolution falls back to the unique cross-script match.
        x = cls()
        assert x.tag() == 1


t = TestPick()
t.test_pick()
t.test_unrecognized()
t.test_bare()
