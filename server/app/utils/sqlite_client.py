import sqlite3
from sqlite3 import Connection
from server.constant.constants import SQLITE_DB_DIR, SQLITE_DB_NAME


def get_db_connection() -> Connection:
    """
    Establishes and returns a connection to the SQLite database.

    Returns:
        Connection: A connection to the SQLite database.
    """
    conn = sqlite3.connect(f"{SQLITE_DB_DIR}/{SQLITE_DB_NAME}")
    # Set row factory to access columns by name
    conn.row_factory = sqlite3.Row
    # Enable WAL mode for better concurrency
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn
