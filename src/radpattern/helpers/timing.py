#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import time
from functools import wraps

def debug_timer(logger=None, name=None):
    """
    Time a function only when DEBUG logging is enabled.

    Usage:
        @debug_timer(log)
        def my_function(...):
            ...
    """

    def decorator(func):
        timer_name = name or func.__qualname__

        @wraps(func)
        def wrapper(*args, **kwargs):
            active_logger = logger or logging.getLogger(func.__module__)

            if not active_logger.isEnabledFor(logging.DEBUG):
                return func(*args, **kwargs)

            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                active_logger.debug("%s took %.6f s", timer_name, elapsed)

        return wrapper

    return decorator
