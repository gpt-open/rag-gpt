# coding=utf-8
from diskcache import Cache
from contextlib import contextmanager


class DiskcacheClient:

    def __init__(self, diskcache_dir):
        self.cache = Cache(diskcache_dir)

    def set(self, key, value, ttl=None):
        """ Set key/value with an optional time-to-live (ttl) """
        self.cache.set(key, value, expire=ttl)

    def get(self, key):
        """ Get value of key; return None if expired or not found """
        return self.cache.get(key, default=None)

    def delete(self, key):
        """ Delete the specified key """
        self.cache.delete(key)

    def append_to_list(self, key, value, ttl=None, max_length=5):
        """ Append an element to a list while keeping the list length not exceeding the maximum """
        with self.cache.transact():
            lst = list(self.cache.get(key, default=[]))
            lst.append(value)
            # Keep the list length not exceeding max_length
            lst = lst[-max_length:]
            self.cache.set(key, lst, expire=ttl)

    def get_list(self, key):
        """ Get the entire list """
        return self.cache.get(key, default=[])

    def expire(self, key, ttl):
        """ Set an expiration time for the key """
        self.cache.expire(key, ttl)
