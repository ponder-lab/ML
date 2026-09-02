# Spike fixture for wala/ML#868 layer two (not for commit): a pytest-shaped test
# method whose only parameter supplier is the decorator's argument list.
import probe868mod


def params(*classes):
    def wrap(f):
        def run(receiver):
            for c in classes:
                if c is not None:
                    f(receiver, c)

        return run

    return wrap


class TestPick:
    @params(probe868mod.A, probe868mod.B, None)
    def test_pick(self, cls):
        x = cls()
        assert x.tag() in (1, 2)


TestPick().test_pick()
