# coding=utf-8
import json
import os
import sqlite3
import time
from werkzeug.security import generate_password_hash
from server.app.utils.diskcache_client import diskcache_client
from server.constant.constants import SQLITE_DB_DIR, SQLITE_DB_NAME
from dotenv import load_dotenv


os.makedirs(SQLITE_DB_DIR, exist_ok=True)


def init_chroma_db():
    # Load environment variables from .env file
    load_dotenv(override=True)
    try:
        from server.constant.env_constants import check_env_variables
        check_env_variables()
        from server.rag.index.embedder.document_embedder import document_embedder
        return True
    except Exception as e:
        print(f"[ERROR] init_chroma_db is failed, the exception is {e}")
        return False


def create_table():
    conn = sqlite3.connect(f'{SQLITE_DB_DIR}/{SQLITE_DB_NAME}')
    cur = conn.cursor()

    # Create table to store domain information and status of sitemap
    cur.execute('''
    CREATE TABLE IF NOT EXISTS t_sitemap_domain_tab (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        domain TEXT NOT NULL,
        domain_status INTEGER NOT NULL,
        version INTEGER NOT NULL,
        ctime INTEGER NOT NULL,
        mtime INTEGER NOT NULL
    )
    ''')
    #`domain_status` meanings:
    #  1 - 'Domain statistics gathering'
    #  2 - 'Domain statistics gathering collected'
    #  3 - 'Domain processing'
    #  4 - 'Domain processed'


    # Create table to store sitemap webpage information
    cur.execute('''
    CREATE TABLE IF NOT EXISTS t_sitemap_url_tab (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        domain TEXT NOT NULL,
        url TEXT NOT NULL,
        content TEXT NOT NULL,
        content_length INTEGER NOT NULL,
        content_md5 TEXT NOT NULL,
        doc_status INTEGER NOT NULL,
        version INTEGER NOT NULL,
        ctime INTEGER NOT NULL,
        mtime INTEGER NOT NULL
    )
    ''')
    #`doc_status` meanings:
    #  0 - 'Process failed'
    #  1 - 'Sitemaps web page recorded'
    #  2 - 'Sitemaps web page crawling'
    #  3 - 'Sitemaps web page crawling completed'
    #  4 - 'Sitemaps web text Embedding stored in VectorDB'
    #  5 - 'Sitemaps web page expired and needed crawled again'


    # Create table to store isolated webpage information
    cur.execute('''
    CREATE TABLE IF NOT EXISTS t_isolated_url_tab (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        url TEXT NOT NULL,
        content TEXT NOT NULL,
        content_length INTEGER NOT NULL,
        content_md5 TEXT NOT NULL,
        doc_status INTEGER NOT NULL,
        ctime INTEGER NOT NULL,
        mtime INTEGER NOT NULL
    )
    ''')
    #`doc_status` meanings:
    #  0 - 'Process failed'
    #  1 - 'Isolated web page recorded'
    #  2 - 'Isolated web page crawling'
    #  3 - 'Isolated web page crawling completed'
    #  4 - 'Isolated web text Embedding stored in VectorDB'


    # Create table to store local file
    cur.execute('''
    CREATE TABLE IF NOT EXISTS t_local_file_tab (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        url TEXT NOT NULL,
        origin_file_name TEXT NOT NULL,
        file_path TEXT NOT NULL,
        file_type TEXT NOT NULL,
        content_length INTEGER NOT NULL,
        content_md5 TEXT NOT NULL,
        doc_status INTEGER NOT NULL,
        ctime INTEGER NOT NULL,
        mtime INTEGER NOT NULL
    )
    ''')
    #`doc_status` meanings:
    #  0 - 'Process failed'
    #  1 - 'Local files recorded'
    #  2 - 'Local files parsing'
    #  3 - 'Local files parsing completed'
    #  4 - 'Local files text Embedding stored in VectorDB'


    cur.execute('''
    CREATE TABLE IF NOT EXISTS t_local_file_chunk_tab (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_id INTEGER NOT NULL,
        chunk_index INTEGER NOT NULL,
        content TEXT NOT NULL,
        content_length INTEGER NOT NULL,
        ctime INTEGER NOT NULL,
        mtime INTEGER NOT NULL
    )
    ''')


    # Create document embedding map table to link documents to their embeddings
    cur.execute('''
    CREATE TABLE IF NOT EXISTS t_doc_embedding_map_tab (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        doc_id INTEGER NOT NULL,
        doc_source INTEGER NOT NULL,
        embedding_id_list TEXT NOT NULL,
        ctime INTEGER NOT NULL,
        mtime INTEGER NOT NULL
    )
    ''')
    #`doc_source` meanings:
    #  1 - 'from sitemap URLs'
    #  2 - 'from isolated URLs'
    #  3 - 'from local files'


    # Create user QA record table to store user queries and responses
    cur.execute('''
    CREATE TABLE IF NOT EXISTS t_user_qa_record_tab (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        query TEXT NOT NULL,
        answer TEXT NOT NULL,
        source TEXT NOT NULL,
        ctime INTEGER NOT NULL,
        mtime INTEGER NOT NULL
    )
    ''')


    # Create user QA intervene table for manual intervention in QA pairs
    cur.execute('''
    CREATE TABLE IF NOT EXISTS t_user_qa_intervene_tab (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        query TEXT NOT NULL,
        intervene_answer TEXT NOT NULL,
        source TEXT NOT NULL,
        ctime INTEGER NOT NULL,
        mtime INTEGER NOT NULL
    )
    ''')


    # Create account table to store user account information
    cur.execute('''
    CREATE TABLE IF NOT EXISTS t_account_tab (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        account_name TEXT NOT NULL,
        password_hash TEXT NOT NULL,
        is_login INTEGER NOT NULL,
        ctime INTEGER NOT NULL,
        mtime INTEGER NOT NULL
    )
    ''')


    # Create bot setting table to store chatbot settings
    cur.execute('''
    CREATE TABLE IF NOT EXISTS t_bot_setting_tab (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        initial_messages TEXT NOT NULL,
        suggested_messages TEXT NOT NULL,
        bot_name TEXT NOT NULL,
        bot_avatar TEXT NOT NULL,
        chat_icon TEXT NOT NULL,
        placeholder TEXT NOT NULL,
        model TEXT NOT NULL,
        ctime INTEGER NOT NULL,
        mtime INTEGER NOT NULL
    )
    ''')

    conn.commit()
    conn.close()


def create_index():
    with sqlite3.connect(f'{SQLITE_DB_DIR}/{SQLITE_DB_NAME}') as conn:
        # the index of t_sitemap_domain_tab
        conn.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_domain ON t_sitemap_domain_tab (domain)')

        # the index of t_sitemap_url_tab
        conn.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_url ON t_sitemap_url_tab (url)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_ctime ON t_sitemap_url_tab (ctime)')

        # the index of t_isolated_url_tab
        conn.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_url ON t_isolated_url_tab (url)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_ctime ON t_isolated_url_tab (ctime)')

        # the index of t_local_file_tab
        conn.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_url ON t_local_file_tab (url)')
        conn.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_content_md5 ON t_local_file_tab (content_md5)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_ctime ON t_local_file_tab (ctime)')

        # the index of t_local_file_chunk_tab
        conn.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_file_id_chunk_index ON t_local_file_chunk_tab (file_id, chunk_index)')

        # the index of t_doc_embedding_map_tab
        conn.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_doc_id_doc_source ON t_doc_embedding_map_tab (doc_id, doc_source)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_ctime ON t_doc_embedding_map_tab (ctime)')

        # the index of t_user_qa_record_tab
        conn.execute('CREATE INDEX IF NOT EXISTS idx_user_id ON t_user_qa_record_tab (user_id)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_ctime ON t_user_qa_record_tab (ctime)')

        # the index of t_user_qa_intervene_tab
        conn.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_query ON t_user_qa_intervene_tab (query)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_ctime ON t_user_qa_intervene_tab (ctime)')

        # the index of t_account_tab
        conn.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_account_name ON t_account_tab (account_name)')


def init_admin_account():
    # Initialize admin account with predefined credentials
    account_name = 'admin'
    password = 'open_kf_AIGC@2024'
    password_hash = generate_password_hash(password, method='pbkdf2:sha256', salt_length=10)

    conn = None
    try:
        conn = sqlite3.connect(f'{SQLITE_DB_DIR}/{SQLITE_DB_NAME}')
        cur = conn.cursor()

        # Check if the account name already exists
        cur.execute('SELECT id FROM t_account_tab WHERE account_name = ?', (account_name,))
        account = cur.fetchone()
        if account:
            print(f"[INFO] account_name:'{account_name}' already exists.")
        else:
            timestamp = int(time.time())
            cur.execute('INSERT INTO t_account_tab (account_name, password_hash, is_login, ctime, mtime) VALUES (?, ?, ?, ?, ?)',
                        (account_name, password_hash, 0, timestamp, timestamp))
            conn.commit()
    except Exception as e:
        print(f"[ERROR] init_admin_account is failed, the exception is {e}")
    finally:
        if conn:
            conn.close()


def init_bot_setting():
    # Initialize bot settings
    initial_messages = ['Hi! What can I help you with?']
    suggested_messages = []
    bot_name = ''
    bot_avatar = ''
    chat_icon = ''
    placeholder = 'Message...'
    model = 'gpt-3.5-turbo'

    conn = None
    try:
        conn = sqlite3.connect(f'{SQLITE_DB_DIR}/{SQLITE_DB_NAME}')
        cur = conn.cursor()

        # Check if the setting table is empty
        cur.execute('SELECT COUNT(*) FROM t_bot_setting_tab')
        if cur.fetchone()[0] > 0:
            print("[INFO] the bot setting already exists.")
        else:
            timestamp = int(time.time())
            cur.execute('''
                INSERT INTO t_bot_setting_tab (id, initial_messages, suggested_messages, bot_name, bot_avatar, chat_icon, placeholder, model, ctime, mtime)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                1,
                json.dumps(initial_messages),
                json.dumps(suggested_messages),
                bot_name, bot_avatar, chat_icon, placeholder, model,
                timestamp, timestamp)
            )
            conn.commit()

            # Add bot setting into Cache
            try:
                key = "open_kf:bot_setting"
                bot_setting = {
                    'id': 1,
                    'initial_messages': initial_messages,
                    'suggested_messages': suggested_messages,
                    'bot_name': bot_name,
                    'bot_avatar': bot_avatar,
                    'chat_icon': chat_icon,
                    'placeholder': placeholder,
                    'model': model,
                    'ctime': timestamp,
                    'mtime': timestamp
                }
                diskcache_client.set(key, json.dumps(bot_setting))
            except Exception as e:
                print(f"[ERROR] add bot setting into Cache is failed, the exception is {e}")
    except Exception as e:
        print(f"[ERROR] init_bot_setting is failed, the exception is {e}")
    finally:
        if conn:
            conn.close()


if __name__ == '__main__':
    print('Create tables in the SQLite database')
    create_table()
    print('Create indexes for the tables')
    create_index()
    print('Initialize the admin account')
    init_admin_account()
    print('Initialize the bot settings')
    init_bot_setting()
    print('SQLite init Done!\n\n')


    print("Init Chroma DB")
    ret = init_chroma_db()
    if ret:
        print("Init Chroma DB Done!")
    else:
        print("Init Chroma DB Failed!")
