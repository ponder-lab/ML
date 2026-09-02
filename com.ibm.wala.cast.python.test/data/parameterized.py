# A faithful local stand-in for absl.testing.parameterized's `parameters`
# decorator (wala/ML#868): each decorator argument becomes one invocation of the
# decorated test method. Vendored so the fixture runs without absl; the analysis
# recognizes the decorator by its mined name, so the site spelling matches the
# real library's.
def parameters(*args):
    def wrap(f):
        def run(receiver):
            for a in args:
                f(receiver, a)

        return run

    return wrap
