# coding=utf-8
import os
import redis

class RedisConfig:
    HOST = os.getenv('REDIS_HOST', '127.0.0.1')  # Use environment variable or default to localhost
    PORT = int(os.getenv('REDIS_PORT', 6379))    # Use environment variable or default to 6379
    DB = int(os.getenv('REDIS_DB', 0))           # Use environment variable or default to DB 0
    SOCKET_TIMEOUT = 5                          # Socket timeout in seconds for read/write operations

# Create a Redis connection pool
redis_pool = redis.ConnectionPool(
    host=RedisConfig.HOST,
    port=RedisConfig.PORT,
    db=RedisConfig.DB,
    socket_timeout=RedisConfig.SOCKET_TIMEOUT
)

redis_client = redis.StrictRedis(connection_pool=redis_pool, decode_responses=True)
