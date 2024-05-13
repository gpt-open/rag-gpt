from contextlib import contextmanager
from typing import Generator, Any
from diskcache import Cache, Lock
from server.app.utils.diskcache_client import diskcache_client
from server.constant.constants import DISTRIBUTED_LOCK_ID, DISTRIBUTED_LOCK_EXPIRE_TIME


class DiskcacheLock:
    def __init__(self,
        cache: Cache,
        lock_id: str,
        expire_time: int = DISTRIBUTED_LOCK_EXPIRE_TIME
    ) -> None:
        """
        Initialize the DiskcacheLock with a Diskcache client and an optional expiration time.

        Args:
            cache (Cache): The diskcache Cache object.
            lock_id (str): A unique string identifier for the lock.
            expire_time (int): The expiration time for the lock in seconds.
        """
        self.cache: Cache = cache
        self.lock_id: str = lock_id
        self.expire_time: int = expire_time

    @contextmanager
    def lock(self) -> Generator[None, None, None]:
        """
        Context manager for acquiring and automatically releasing a lock.

        Yields:
            Generator[None, None, None]: Yields nothing and holds the lock until the 'with' block is completed.
        """
        with Lock(self.cache, self.lock_id, expire=self.expire_time):
            yield  # Hold the lock until the 'with' block completes


# Initialize Diskcache distributed lock
diskcache_lock = DiskcacheLock(diskcache_client.cache, DISTRIBUTED_LOCK_ID)
