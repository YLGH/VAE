def list_generator(l, size):
    start = 0
    while start + size < len(l):
        yield l[start:start + size]
        start += size
    yield l[start:]
