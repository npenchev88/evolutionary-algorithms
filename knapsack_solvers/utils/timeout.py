import signal


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException


def with_timeout(timeout, func, cname, *args, **kwargs):
    # Set the timeout handler
    signal.signal(signal.SIGALRM, timeout_handler)
    # Set an alarm
    signal.alarm(timeout)
    try:
        result = func(*args, **kwargs)
        signal.alarm(0)  # Disable the alarm after successful execution
        return result
    except TimeoutException:
        print(f"Timeout: {cname}")
        return cname
