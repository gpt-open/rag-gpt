from contextlib import contextmanager
from typing import Any, Optional, List
from diskcache import Cache
from server.constant.constants import DISKCACHE_DIR


class DiskcacheClient:
    def __init__(self, diskcache_dir: str) -> None:
        self.cache: Cache = Cache(diskcache_dir)

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set key/value with an optional time-to-live (ttl).

        Args:
            key (str): The key under which the value is stored.
            value (Any): The value to store.
            ttl (Optional[int]): Optional time-to-live in seconds.
        """
        self.cache.set(key, value, expire=ttl)

    def get(self, key: str) -> Any:
        """
        Get the value of a key; return None if expired or not found.

        Args:
            key (str): The key whose value is to be retrieved.

        Returns:
            Any: The value associated with the key, or None.
        """
        return self.cache.get(key, default=None)

    def delete(self, key: str) -> None:
        """
        Delete the specified key.

        Args:
            key (str): The key to be deleted.
        """
        self.cache.delete(key)

    def append_to_list(self, key: str, value: Any, ttl: Optional[int] = None, max_length: int = 5) -> None:
        """
        Append an element to a list while keeping the list length not exceeding the maximum.

        Args:
            key (str): The key under which the list is stored.
            value (Any): The value to append to the list.
            ttl (Optional[int]): Optional time-to-live in seconds.
            max_length (int): Maximum length of the list after appending.
        """
        with self.cache.transact():
            lst: List[Any] = list(self.cache.get(key, default=[]))
            lst.append(value)
            # Keep the list length not exceeding max_length
            lst = lst[-max_length:]
            self.cache.set(key, lst, expire=ttl)

    def get_list(self, key: str) -> List[Any]:
        """
        Get the entire list.

        Args:
            key (str): The key whose list value is to be retrieved.

        Returns:
            List[Any]: The list stored at the key, or an empty list.
        """
        return self.cache.get(key, default=[])

    def expire(self, key: str, ttl: int) -> None:
        """
        Set an expiration time for the key.

        Args:
            key (str): The key for which to set the expiration.
            ttl (int): Time-to-live in seconds.
        """
        self.cache.expire(key, ttl)


# Initialize Diskcache client
diskcache_client = DiskcacheClient(DISKCACHE_DIR)
