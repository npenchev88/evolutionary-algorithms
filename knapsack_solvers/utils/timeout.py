import signal


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException


def with_timeout(timeout, func, cname, *args, **kwargs):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    try:
        result = func(*args, **kwargs)
        signal.alarm(0)
        return result
    except TimeoutException:
        print(f"Timeout: {cname}")
        return cname
