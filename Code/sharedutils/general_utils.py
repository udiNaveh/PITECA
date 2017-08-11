
def all_same(iter, f):
    first = iter[0]
    return not any(f(x) != f(first) for x in iter)

def zeropad(i, length):
    assert type(i)==int
    i = str(i)
    n_zeros = length - len(str(i))
    if n_zeros>0:
        i = '0'*n_zeros + i
    return i
