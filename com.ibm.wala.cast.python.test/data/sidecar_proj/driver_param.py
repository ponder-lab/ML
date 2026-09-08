# wala/ML#901 precondition guard: a sidecar entry anchored directly on a function parameter.
# The resolver walks instruction defs, and a parameter has no defining instruction, so this anchor
# matches nothing and `image` stays untyped. `testTypeAnnotationCannotAnchorOnParameter` pins that:
# if the anchor grammar ever gains parameter support, `image` gets seeded and the test fails,
# flagging the now-stale safety comment at the parameter-origin stamp in PythonTensorAnalysisEngine.
def consume(t):
    pass


def transform(image):
    consume(image)


def load_untyped():
    return 0


transform(load_untyped())
