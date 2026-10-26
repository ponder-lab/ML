# A keyword argument binds the callee's parameter of that name and stores nothing in the constant
# naming the keyword (wala/ML#1151).


def f(learning_rate):
    return learning_rate


def g(learning_rate):
    return learning_rate


assert f(learning_rate=[2]) == [2]
assert g(learning_rate=0.1) == 0.1
