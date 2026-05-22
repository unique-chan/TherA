from contextlib import contextmanager

_IP_TOK = None

@contextmanager
def ip_tokens_context(x):
    global _IP_TOK
    old = _IP_TOK
    _IP_TOK = x
    try:
        yield
    finally:
        _IP_TOK = old

def current_ip_tokens():
    return _IP_TOK


