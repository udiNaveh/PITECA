
def all_same(iter, f):
    first = iter[0]
    return not any(f(x) != f(first) for x in iter)
