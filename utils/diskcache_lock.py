# coding=utf-8
from diskcache import Lock
from contextlib import contextmanager


class DiskcacheLock:

    def __init__(self, cache, lock_id, expire_time=10):
        """Initialize the DiskcacheLock with a Diskcache client and an optional expiration time."""
        self.cache = cache
        self.lock_id = lock_id
        self.expire_time = expire_time

    @contextmanager
    def lock(self):
        """Context manager for acquiring and automatically releasing a lock."""
        with Lock(self.cache, self.lock_id, expire=self.expire_time):
            yield  # Hold the lock until the 'with' block completes
