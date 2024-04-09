# coding=utf-8
import asyncio
import uuid

class RedisLock:

    def __init__(self, redis_client, lock_id, expire_time=10):
        """Initialize the RedisLock with a Redis client, lock ID, and an optional expiration time."""
        self.redis_client = redis_client
        self.lock_id = lock_id
        self.expire_time = expire_time
        self.lock_value = str(uuid.uuid4())

    async def aacquire_lock(self):
        """Asynchronously wrap the synchronous method of acquiring the lock."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.acquire_lock)

    def acquire_lock(self):
        """The actual synchronous method to acquire the lock."""
        # The 'nx=True' argument means "SET if Not eXists"
        # The 'ex=self.expire_time' argument sets the expiration time for the lock
        return self.redis_client.set(self.lock_id, self.lock_value, nx=True, ex=self.expire_time)

    async def arelease_lock(self):
        """Asynchronously wrap the synchronous method of releasing the lock."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.release_lock)

    def release_lock(self):
        """The actual synchronous method to release the lock."""
        # This Lua script checks if the lock is held by this instance before deleting it.
        # 'KEYS[1]' is the lock key, 'ARGV[1]' is the lock value.
        script = """
        if redis.call("get", KEYS[1]) == ARGV[1] then
            return redis.call("del", KEYS[1])
        else
            return 0
        end
        """
        # Executes the Lua script, passing the lock ID and lock value
        return self.redis_client.eval(script, 1, self.lock_id, self.lock_value)

