# decorator function to type and range check parameters
def paramcheck(types, ranges=None):
    def __f(f):
        def _f(*args, **kwargs):
            for a, t in zip(args, types):
                if not isinstance(a, t):
                    raise TypeError("expected type %s got %r" % (t, a))
            for a, r in zip(args, ranges or []):
                if r and not r[0] <= a <= r[1]:
                    raise ValueError("expected value in range %r: %r" % (r, a))
            return f(*args, **kwargs)
        return _f
    return __f


def methodcheck(types, ranges=None):
    def __f(f):
        def _f(*args, **kwargs):
            for a, t in zip(args[1:], types):
                if not isinstance(a, t):
                    raise TypeError("method expected type %s got %r" % (t, a))
            for a, r in zip(args[1:], ranges or []):
                if r and not r[0] <= a <= r[1]:
                    raise ValueError("method expected value in range %r: %r" % (r, a))
            return f(*args, **kwargs)
        return _f
    return __f

