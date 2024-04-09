# coding=utf-8
import redis

class RedisConfig:
    HOST = '127.0.0.1'  # Redis server host
    PORT = 6379         # Redis server port
    DB = 0              # Default database to use
    SOCKET_TIMEOUT = 5  # Socket timeout in seconds for read/write operations

# Create a Redis connection pool
redis_pool = redis.ConnectionPool(
    host=RedisConfig.HOST,
    port=RedisConfig.PORT,
    db=RedisConfig.DB,
    socket_timeout=RedisConfig.SOCKET_TIMEOUT
)

redis_client = redis.StrictRedis(connection_pool=redis_pool, decode_responses=True)
