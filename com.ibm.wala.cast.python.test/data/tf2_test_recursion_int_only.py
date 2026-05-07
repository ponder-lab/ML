def recursive_fn(n):
    if n > 0:
        return recursive_fn(n - 1)
    return 1


recursive_fn(5)
