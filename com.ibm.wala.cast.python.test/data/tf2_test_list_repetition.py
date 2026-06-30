def consume(value):
    pass


# `[0] * 3` is Python list repetition (produces a list), not tensor scalar-multiplication. `value`
# should not be typed as a tensor. wala/ML#653.
consume([0] * 3)
