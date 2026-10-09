# A module under a package directory: its function reads `print` and `isinstance` from the module
# scope, where the builtins are bound, and each call must reach its builtin.
def f(x):
    if isinstance(x, int):
        print("Traced with: " + str(x))
    return x


f(1)
