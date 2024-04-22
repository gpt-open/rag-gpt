# coding=utf-8
import asyncio
from functools import wraps
import json
import os
import sqlite3
from threading import Thread
import time
from urllib.parse import urljoin, urlparse
import uuid
from dotenv import load_dotenv
from flask import Flask, request, jsonify, Response, send_from_directory, abort
from flask_cors import CORS
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from utils.diskcache_config import DiskcacheClient
from utils.diskcache_lock import DiskcacheLock
from utils.token_helper import TokenHelper
from utils.logger_config import my_logger as logger
from crawler_module.web_link_crawler import AsyncCrawlerSiteLink
from crawler_module.web_content_crawler import AsyncCrawlerSiteContent
from crawler_module.document_embedding import DocumentEmbedder


# Load environment variables from .env file
load_dotenv(override=True)

# Retrieve environment variables or set default values
DISKCACHE_DIR = os.getenv('DISKCACHE_DIR', 'your_diskcache_directory')
SQLITE_DB_DIR = os.getenv('SQLITE_DB_DIR', 'your_sqlite_db_directory')
SQLITE_DB_NAME = os.getenv('SQLITE_DB_NAME', 'your_sqlite_db_name')
MAX_CRAWL_PARALLEL_REQUEST = int(os.getenv('MAX_CRAWL_PARALLEL_REQUEST', '5'))
CHROMA_DB_DIR = os.getenv('CHROMA_DB_DIR', 'your_chroma_db_directory')
CHROMA_COLLECTION_NAME = os.getenv('CHROMA_COLLECTION_NAME', 'your_collection_name')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'your_openai_api_key')
GPT_MODEL_NAME = os.getenv('GPT_MODEL_NAME', 'gpt-3.5-turbo')
OPENAI_EMBEDDING_MODEL_NAME = os.getenv('OPENAI_EMBEDDING_MODEL_NAME', 'text-embedding-3-small')
MAX_CHUNK_LENGTH = int(os.getenv('MAX_CHUNK_LENGTH', '1300'))
MAX_QUERY_LENGTH = int(os.getenv('MAX_QUERY_LENGTH', '200'))
RECALL_TOP_K = int(os.getenv('RECALL_TOP_K', '3'))
MIN_RELEVANCE_SCORE = float(os.getenv('MIN_RELEVANCE_SCORE', '0.5'))
MAX_HISTORY_SESSION_LENGTH = int(os.getenv('MAX_HISTORY_SESSION_LENGTH', '3'))
SESSION_EXPIRE_TIME = int(os.getenv('SESSION_EXPIRE_TIME', '10800'))
SITE_TITLE = os.getenv('SITE_TITLE', 'your_site_title')
STATIC_DIR = os.getenv('STATIC_DIR', 'your_static_dir')
URL_PREFIX = os.getenv('URL_PREFIX', 'your_url_prefix')
MEDIA_DIR = os.getenv('MEDIA_DIR', 'your_media_dir')


app = Flask(__name__, static_folder=STATIC_DIR)
CORS(app)


# Initialize Diskcache client
g_diskcache_client = DiskcacheClient(DISKCACHE_DIR)

# Initialize Diskcache distributed lock
g_diskcache_lock = DiskcacheLock(g_diskcache_client.cache, 'open_kf:distributed_lock')

# Set OpenAI GPT API key
g_client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize OpenAI embeddings with the specified model
g_embeddings = OpenAIEmbeddings(
    model=OPENAI_EMBEDDING_MODEL_NAME,
    openai_api_key=OPENAI_API_KEY
)

g_document_embedder = DocumentEmbedder(
    collection_name=CHROMA_COLLECTION_NAME,
    embedding_function=g_embeddings,
    persist_directory=CHROMA_DB_DIR
)

def get_db_connection():
    conn = sqlite3.connect(f"{SQLITE_DB_DIR}/{SQLITE_DB_NAME}")
    conn.row_factory = sqlite3.Row              # Set row factory to access columns by name
    conn.execute("PRAGMA journal_mode=WAL;")    # Enable WAL mode for better concurrency
    return conn


"""
Background:
In scenarios where using a dedicated static file server (like Nginx) is not feasible or desired, Flask can be configured to serve static files directly. This setup is particularly useful during development or in lightweight production environments where simplicity is preferred over the scalability provided by dedicated static file servers.

This Flask application demonstrates how to serve:
- Static media files from a dynamic path (`MEDIA_DIR`)
- The homepages and assets for two single-page applications (SPAs): 'open-kf-chatbot' and 'open-kf-admin'.

Note:
While Flask is capable of serving static files, it's not optimized for the high performance and efficiency of a dedicated static file server like Nginx, especially under heavy load. For large-scale production use cases, deploying a dedicated static file server is recommended.

The provided routes include a dynamic route for serving files from a specified media directory and specific routes for SPA entry points and assets. This configuration ensures that SPA routing works correctly without a separate web server.
"""
# Dynamically serve files from the MEDIA_DIR
@app.route(f'/{MEDIA_DIR}/<filename>')
def serve_media_file(filename):
    # Construct the complete file path within the MEDIA_DIR
    file_path = os.path.join(app.static_folder, MEDIA_DIR, filename)
    # Check if the file exists and serve it if so
    if os.path.exists(file_path) and os.path.isfile(file_path):
        return send_from_directory(os.path.join(app.static_folder, MEDIA_DIR), filename)
    else:
        # Return a 404 error if the file does not exist
        abort(404)

# Route for the homepage of the 'open-kf-chatbot' site
@app.route('/open-kf-chatbot', strict_slashes=False)
def index_chatbot():
    return send_from_directory(f'{app.static_folder}/open-kf-chatbot', 'index.html')

# Route for serving other static files of the 'open-kf-chatbot' application
@app.route('/open-kf-chatbot/<path:path>')
def serve_static_chatbot(path):
    return send_from_directory(f'{app.static_folder}/open-kf-chatbot', path)

# Route for the homepage of the 'open-kf-admin' site
@app.route('/open-kf-admin', strict_slashes=False)
def index_admin():
    return send_from_directory(f'{app.static_folder}/open-kf-admin', 'index.html')

# Route for serving other static files of the 'open-kf-admin' application
@app.route('/open-kf-admin/<path:path>')
def serve_static_admin(path):
    return send_from_directory(f'{app.static_folder}/open-kf-admin', path)


def token_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = None
        if 'Authorization' in request.headers:
            token = request.headers['Authorization'].split(" ")[1]
        if not token:
            logger.error("Token is missing!")
            return jsonify({'retcode': -10000, 'message': 'Token is missing!', 'data': {}}), 401
        try:
            user_payload = TokenHelper.verify_token(token)
            if user_payload == 'Token expired':
                logger.error(f"Token: '{token}' is expired!")
                return jsonify({'retcode': -10001, 'message': 'Token is expired!', 'data': {}}), 401
            elif user_payload == 'Invalid token':
                logger.error(f"Token: '{token}' is invalid")
                return jsonify({'retcode': -10002, 'message': 'Token is invalid!', 'data': {}}), 401
            request.user_payload = user_payload  # Store payload in request for further use
        except Exception as e:
            logger.error(f"Token: '{token}' is invalid, the exception is {e}")
            return jsonify({'retcode': -10003, 'message': 'Token is invalid!', 'data': {}}), 401
        return f(*args, **kwargs)
    return decorated_function

@app.route('/open_kf_api/get_token', methods=['POST'])
def get_token():
    data = request.json
    user_id = data.get('user_id')
    if not user_id:
        return jsonify({'retcode': -20000, 'message': 'user_id is required', 'data': {}})

    try:
        # generate token
        token = TokenHelper.generate_token(user_id)
        logger.success(f"generate token:'{token}' with user_id:'{user_id}'")
        return jsonify({"retcode": 0, "message": "success", "data": {"token": token}})
    except Exception as e:
        logger.error(f"generate token with user_id:'{user_id}' is failed, the exception is {e}")
        return jsonify({'retcode': -20001, 'message': str(e), 'data': {}})


def get_user_query_history(user_id):
    history_key = f"open_kf:query_history:{user_id}"
    history_items = g_diskcache_client.get_list(history_key)[::-1]
    history = [json.loads(item) for item in history_items]
    return history

def preprocess_query(query, site_title):
    # Convert to lowercase for case-insensitive comparison
    query_lower = query.lower()
    site_title_lower = site_title.lower()
    # Check if the site title is already included in the query
    if site_title_lower not in query_lower:
        adjust_query =  f"{query}\t{site_title}"
        logger.warning(f"adjust_query:'{adjust_query}'")
        return adjust_query
    return query

def search_and_answer(query, user_id, k=RECALL_TOP_K, is_streaming=False):
    site_title = SITE_TITLE
    # Perform similarity search
    beg = time.time()
    adjust_query = preprocess_query(query, site_title)
    results = g_document_embedder.search_document(adjust_query, k)
    timecost = time.time() - beg

    unfiltered_context = "\n--------------------\n".join([
        f"URL: {doc.metadata['source']}\nRevelance score: {score}\nContent: {doc.page_content}"
        for doc, score in results
    ])
    logger.info(f"for the query:'{query}' and user_id:'{user_id}'\nunfiltered_context={unfiltered_context}\nthe timecost of search_document is {timecost}")

    # Get history session from Cache
    history_session = get_user_query_history(user_id)
    # Build the history_context for prompt
    history_context = "\n--------------------\n".join([f"Previous Query: {item['query']}\nPrevious Answer: {item['answer']}" for item in history_session])

    # Build the context for prompt
    recall_domain_set = set()
    filtered_results = []
    for doc, score in results:
        if score > MIN_RELEVANCE_SCORE:
            filtered_results.append((doc, score))
            domain = urlparse(doc.metadata['source']).netloc
            recall_domain_set.add(domain)

    if filtered_results:
        filtered_context = "\n--------------------\n".join([
            f"URL: {doc.metadata['source']}\nRevelance score: {score}\nContent: {doc.page_content}"
            for doc, score in filtered_results
        ])
        context = f"""Documents size: {len(filtered_results)}
Documents(Sort by Relevance Score from high to low):
{filtered_context}"""
    else:
        context = f"""Documents size: 0
No documents found directly related to the current query.
If the query is similar to greeting phrases like ['Hi', 'Hello', 'Who are you?'], including greetings in other languages such as Chinese, Russian, French, Spanish, German, Japanese, Arabic, Hindi, Portuguese, Korean, etc. The bot will offer a friendly standard response, guiding users to seek information or services detailed on the `{site_title}` website.
For other queries, please give the answer "Unfortunately, there is no information available about '{query}' on the `{site_title}` website. I'm here to assist you with information related to the `{site_title}` website. If you have any specific queries about our services or need help, feel free to ask, and I'll do my best to provide you with accurate and relevant answers."
"""

    prompt = f"""
This smart customer service bot is designed to provide users with information directly related to the `{site_title}` website's content. It employs a combination of Large Language Model (LLM) and Retriever-Augmented Generation (RAG) technologies to accurately identify the most relevant documents for user queries, thereby ensuring responses are both contextually pertinent and timely.

The bot focuses on queries specifically related to the content of the `{site_title}` website, and will inform users when a query falls outside of this scope. For queries related to the `{site_title}` website, the bot must provide accurate and detailed answers.

If the query is similar to greeting phrases like ['Hi', 'Hello', 'Who are you?'], including greetings in other languages such as Chinese, Russian, French, Spanish, German, Japanese, Arabic, Hindi, Portuguese, Korean, etc. The bot will offer a friendly standard response, guiding users to seek information or services detailed on the `{site_title}` website.

Responses from the bot take into account the user's previous interactions, adapting to their potential interests or previous unanswered queries. It strives not only to provide answers but to offer comprehensive insights, including URLs, steps, example codes, and more, as necessary.

Given the documents listed below and the user's query history, please provide accurate and detailed answers in the query's language. The response should be in JSON, with 'answer' and 'source' fields. Answers must be based on these documents and directly relevant to the `{site_title}` website. If the query is unrelated to the documents, inform the user that answers cannot be provided.

User ID: "{user_id}"
Query: "{query}"

User Query History(Sort by request time from most recent to oldest):
{history_context}

{context}

Response Requirements:
- If unsure about the answer, proactively seek clarification.
- Refer only to knowledge related to `{site_title}` website's content.
- Inform users that queries unrelated to `{site_title}` website's content cannot be answered and encourage them to ask site-related queries.
- Ensure answers are consistent with information on the `{site_title}` website.
- Use Markdown syntax to format the answer for readability.
- Responses must be crafted in the same language as the query.

The most important instruction: Please do not provide any answers unrelated to the `{site_title}` website! Regardless of whether the bot know the answer or not! This instruction must be followed!

Please format your response as follows:
{{
    "answer": "A detailed and specific answer, crafted in the query's language.",
    "source": ["Unique document URLs directly related to the answer. If no documents are referenced, use an empty list []."]
}}

Please format `answer` as follows:
The `answer` must be fully formatted using Markdown syntax to ensure proper rendering in the browser or APP. This includes:
- The `answer` must not be identical to the `query`.
- **Bold** (`**bold**`) and *italic* (`*italic*`) text for emphasis.
- Unordered lists (`- item`) for itemization and ordered lists (`1. item`) for sequencing.
- `Inline code` (`` `Inline code` ``) for brief code snippets and (` ``` `) for longer examples, specifying the programming language for syntax highlighting when possible.
- [Hyperlinks](URL) (`[Hyperlinks](URL)`) to reference external sources.
- Headings (`# Heading 1`, `## Heading 2`, ...) to structure the answer effectively.
"""

    logger.info(f"prompt={prompt}")

    # Call GPT model to generate an answer
    response = g_client.chat.completions.create(
        model=GPT_MODEL_NAME,
        response_format={ "type": "json_object" },
        messages=[{"role": "system", "content": prompt}],
        temperature=0,
        top_p=0.7,
        stream=is_streaming
    )
    logger.warning(f"[Track token consumption] for query='{query}', usage={response.usage}")
    return response, recall_domain_set


def save_user_query_history(user_id, query, answer):
    try:
        answer_json = json.loads(answer)

        # After generating the response from GPT
        # Store user query and GPT response in Cache
        history_key = f"open_kf:query_history:{user_id}"
        history_data = {'query': query, 'answer': answer_json}
        g_diskcache_client.append_to_list(history_key, json.dumps(history_data), ttl=SESSION_EXPIRE_TIME, max_length=MAX_HISTORY_SESSION_LENGTH)
    except Exception as e:
        logger.error(f"query:'{query}' and user_id:'{user_id}' is processed failed with Cache, the exception is {e}")

    timestamp = int(time.time())
    conn = None
    try:
        # Store user query and GPT resposne in DB
        conn = get_db_connection()
        try:
            with g_diskcache_lock.lock():
                conn.execute('INSERT INTO t_user_qa_record_tab (user_id, query, answer, source, ctime, mtime) VALUES (?, ?, ?, ?, ?, ?)',
                             (user_id, query, answer_json["answer"], json.dumps(answer_json["source"]), timestamp, timestamp))
                conn.commit()
        except Exception as e:
            logger.error(f"process g_discache_lock exception:{e}")
    except Exception as e:
        logger.error(f"query:'{query}' and user_id:'{user_id}' is processed failed with Database, the exception is {e}")
    finally:
        if conn:
            conn.close()

def check_smart_query(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        data = request.json
        user_id = data.get('user_id')
        query = data.get('query')
        if not user_id or not query:
            logger.error(f"user_id and query are required")
            return jsonify({'retcode': -20000, 'message': 'user_id and query are required', 'data': {}}), 400
        
        request.user_id = user_id
        request.query = query
        request.intervene_data = None

        try:
            # Check if the query is in Cache
            key = f"open_kf:intervene:{query}"
            intervene_data = g_diskcache_client.get(key)
            if intervene_data:
                logger.info(f"user_id:'{user_id}', query:'{query}' is hit in Cache, the intervene_data is {intervene_data}")
                request.intervene_data = intervene_data
        except Exception as e:
            logger.error(f"Cache exception {e} for user_id:'{user_id}' and query:'{query}'")
        return f(*args, **kwargs)
    return decorated_function

def postprocessing_llm_response(query, answer_json, site_title, recall_domain_set):
    if answer_json['source']:
        answer_json['source'] = list(dict.fromkeys(answer_json['source']))

    is_adjusted = False
    adjust_source = []
    for url in answer_json['source']:
        domain = urlparse(url).netloc
        if domain in recall_domain_set:
            adjust_source.append(url)
        else:
            logger.warning(f"url:'{url}' is not in {recall_domain_set}, it should not be returned!")
            if not is_adjusted:
                is_adjusted = True

    if is_adjusted:
        answer_json['source'] = adjust_source
        logger.warning(f"adjust_source:{adjust_source}")

    if not adjust_source:
        if site_title not in answer_json['answer']:
            adjust_answer = f"Unfortunately, there is no information available about '{query}' on the `{site_title}` website. I'm here to assist you with information related to the `{site_title}` website. If you have any specific queries about our services or need help, feel free to ask, and I'll do my best to provide you with accurate and relevant answers."
            logger.warning(f"adjust_answer:'{adjust_answer}'")
            if not is_adjusted:
                is_adjusted = True
            answer_json['answer'] = adjust_answer
    return is_adjusted

@app.route('/open_kf_api/smart_query', methods=['POST'])
@check_smart_query
@token_required
def smart_query():
    try:
        user_id = request.user_id
        query = request.query
        intervene_data = request.intervene_data
        if intervene_data:
            save_user_query_history(user_id, query, intervene_data)
            intervene_data_json = json.loads(intervene_data)
            return jsonify({"retcode": 0, "message": "success", "data": intervene_data_json})

        if len(query) > MAX_QUERY_LENGTH:
            query = query[:MAX_QUERY_LENGTH]

        #last_character = query[-1]
        #if last_character != "？" and last_character != "?":
        #    query += "?"

        beg = time.time()
        response, recall_domain_set = search_and_answer(query, user_id)
        answer = response.choices[0].message.content

        timecost = time.time() - beg
        answer_json = json.loads(answer)
        answer_json["source"] = list(dict.fromkeys(answer_json["source"]))
        logger.success(f"query:'{query}' and user_id:'{user_id}' is processed successfully, the answer is {answer}\nthe total timecost is {timecost}\n")
        is_adjusted = postprocessing_llm_response(query, answer_json, SITE_TITLE, recall_domain_set)
        if is_adjusted:
            answer = json.dumps(answer_json)
        save_user_query_history(user_id, query, answer)
        return jsonify({"retcode": 0, "message": "success", "data": answer_json})
    except Exception as e:
        logger.error(f"query:'{query}' and user_id:'{user_id}' is processed failed, the exception is {e}")
        return jsonify({'retcode': -20001, 'message': str(e), 'data': {}}), 500


@app.route('/open_kf_api/smart_query_stream', methods=['POST'])
@check_smart_query
@token_required
def smart_query_stream():
    headers = {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'X-Accel-Buffering': 'no'
    }

    try:
        user_id = request.user_id
        query = request.query
        intervene_data = request.intervene_data
        if intervene_data:
            save_user_query_history(user_id, query, intervene_data)

            def generate_intervene():
                yield intervene_data
            return Response(generate_intervene(), mimetype="text/event-stream", headers=headers)

        if len(query) > MAX_QUERY_SIZE:
            query = query[:MAX_QUERY_SIZE]

        #last_character = query[-1]
        #if last_character != "？" and last_character != "?":
        #    query += "?"

        beg = time.time()
        def generate_llm():
            answer_chunks = []
            response = search_and_answer(query, user_id, is_streaming=True)
            for chunk in response:
                logger.info(f"chunk:{chunk}")
                content = chunk.choices[0].delta.content
                if content:
                    answer_chunks.append(content)
                    # Send each answer segment
                    yield content
            # After the streaming response is complete, save to Cache and SQLite
            answer = ''.join(answer_chunks)
            timecost = time.time() - beg
            logger.success(f"query:'{query}' and user_id:'{user_id}' is processed successfully, the answer is {answer}\nthe total timecost is {timecost}\n")
            save_user_query_history(user_id, query, answer)
        return Response(generate_llm(), mimetype="text/event-stream", headers=headers)
    except Exception as e:
        logger.error(f"query:'{query}' and user_id:'{user_id}' is processed failed, the exception is {e}")
        return jsonify({'retcode': -30000, 'message': str(e), 'data': {}}), 500


@app.route('/open_kf_api/get_user_conversation_list', methods=['POST'])
@token_required
def get_user_conversation_list():
    """Retrieve a list of user conversations within a specified time range, with pagination and total count."""
    data = request.json
    start_timestamp = data.get('start_timestamp')
    end_timestamp = data.get('end_timestamp')
    page = data.get('page')
    page_size = data.get('page_size')

    if None in ([start_timestamp, end_timestamp, page, page_size]):
        return jsonify({'retcode': -20000, 'message': 'Missing required parameters'})

    conn = get_db_connection()
    try:
        cur = conn.cursor()
        offset = (page - 1) * page_size

        # First, get the total count of distinct user_ids within the time range for pagination
        cur.execute("""
            SELECT COUNT(DISTINCT user_id) AS total_count FROM t_user_qa_record_tab
            WHERE ctime BETWEEN ? AND ?
        """, (start_timestamp, end_timestamp))
        total_count = cur.fetchone()['total_count']

        # Then, fetch the most recent conversation record for each distinct user within the time range
        cur.execute('''
                        SELECT t.* FROM t_user_qa_record_tab t
                        INNER JOIN (
                                      SELECT user_id, MAX(ctime) AS max_ctime
                                      FROM t_user_qa_record_tab
                                      WHERE ctime BETWEEN ? AND ?
                                      GROUP BY user_id
                                  ) tm ON t.user_id = tm.user_id AND t.ctime = tm.max_ctime
                        ORDER BY t.ctime DESC LIMIT ? OFFSET ?
                    ''', (start_timestamp, end_timestamp, page_size, offset))
        conversation_list = [{
            "user_id": row["user_id"],
            "latest_query": {
                "id": row["id"],
                "query": row["query"],
                "answer": row["answer"],
                "source": json.loads(row["source"]),
                "ctime": row["ctime"],
                "mtime": row["mtime"]
            }
        } for row in cur.fetchall()]

        return jsonify({'retcode': 0, 'message': 'Success', 'data': {'total_count': total_count, 'conversation_list': conversation_list}})
    except Exception as e:
        logger.error(f"Failed to retrieve user conversation list: {e}")
        return jsonify({'retcode': -30000, 'message': 'Internal server error'})
    finally:
        if conn:
            conn.close()


@app.route('/open_kf_api/get_user_query_history_list', methods=['POST'])
@token_required
def get_user_query_history_list():
    data = request.json
    page = data.get('page')
    page_size = data.get('page_size')
    user_id = data.get('user_id')

    # Check for mandatory parameters
    if None in (page, page_size, user_id):
        logger.error("page, page_size and user_id are required")
        return jsonify({'retcode': -20000, 'message': 'page, page_size and user_id are required', 'data': {}})

    try:
        # Convert timestamps and pagination parameters to integers
        page = int(page)
        page_size = int(page_size)
    except ValueError as e:
        logger.error(f"Parameter conversion error: {e}")
        return jsonify({'retcode': -20001, 'message': 'Invalid parameters', 'data': {}})

    # Build query conditions
    query_conditions = "WHERE user_id = ?"
    params = [user_id]

    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # First, query the total count of records under the given conditions
        cur.execute(f'SELECT COUNT(*) FROM t_user_qa_record_tab {query_conditions}', params)
        total_count = cur.fetchone()[0]

        # Then, query the paginated records
        cur.execute(f'SELECT * FROM t_user_qa_record_tab {query_conditions} ORDER BY id LIMIT ? OFFSET ?',
                    params + [page_size, (page-1) * page_size])
        rows = cur.fetchall()

        record_list = [dict(row) for row in rows]  # Convert rows to dictionaries
        # Apply json.loads on the 'source' field of each record
        for record in record_list:
            if 'source' in record:  # Ensure the 'source' key exists
                try:
                    record['source'] = json.loads(record['source'])  # Convert JSON string to Python list
                except json.JSONDecodeError:
                    record['source'] = []  # If decoding fails, set to an empty list or other default value

        return jsonify({
            "retcode": 0,
            "message": "success",
            "data": {
                "total_count": total_count,
                "query_list": record_list
            }
        })
    except Exception as e:
        logger.error(f"Database exception: {e}")
        return jsonify({'retcode': -30000, 'message': 'Database exception', 'data': {}})
    finally:
        if conn:
            conn.close()


@app.route('/open_kf_api/add_intervene_record', methods=['POST'])
@token_required
def add_intervene_record():
    data = request.json
    query = data.get('query')
    intervene_answer = data.get('intervene_answer')
    source = data.get('source', [])

    if None in (query, intervene_answer, source):
        return jsonify({'retcode': -20000, 'message': 'Missing mandatory parameters', 'data': {}})

    conn = None
    try:
        # Check if query already exists in the database
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute('SELECT COUNT(*) FROM t_user_qa_intervene_tab WHERE query = ?', (query,))
        result = cur.fetchone()
        if result and result[0] > 0:
            logger.error(f"intervene query:'{query}' is already exists in the database")
            return jsonify({'retcode': -30000, 'message': 'Query already exists in the database', 'data': {}})

        # Insert the intervene record into DB
        timestamp = int(time.time())
        source_str = json.dumps(source)

        try:
            with g_diskcache_lock.lock():
                cur.execute('INSERT INTO t_user_qa_intervene_tab (query, intervene_answer, source, ctime, mtime) VALUES (?, ?, ?, ?, ?)',
                            (query, intervene_answer, source_str, timestamp, timestamp))
                conn.commit()
        except Exception as e:
            logger.error(f"process g_discache_lock exception:{e}")

        # Update Cache using simple string with the query as the key (prefixed)
        key = f"open_kf:intervene:{query}"
        value = json.dumps({"answer": intervene_answer, "source": source})
        g_diskcache_client.set(key, value)

        return jsonify({"retcode": 0, "message": "success", 'data': {}})
    except Exception as e:
        return jsonify({'retcode': -30000, 'message': 'Database or Cache error', 'data': {}})
    finally:
        if conn:
            conn.close()


@app.route('/open_kf_api/delete_intervene_record', methods=['POST'])
@token_required
def delete_intervene_record():
    data = request.json
    record_id = data.get('id')

    if not record_id:
        return jsonify({'retcode': -20000, 'message': 'id is required', 'data': {}})

    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # First, find the query string for the given id to delete it from Cache
        cur.execute('SELECT query FROM t_user_qa_intervene_tab WHERE id = ?', (record_id,))
        row = cur.fetchone()

        if row:
            query = row['query']
            # Delete the record from DB
            try:
                with g_diskcache_lock.lock():
                    cur.execute('DELETE FROM t_user_qa_intervene_tab WHERE id = ?', (record_id,))
                    conn.commit()
            except Exception as e:
                logger.error(f"process g_discache_lock exception:{e}")

            # Now, delete the corresponding record from Cache
            key = f"open_kf:intervene:{query}"
            g_diskcache_client.delete(key)

            return jsonify({"retcode": 0, "message": "success", 'data': {}})
        else:
            return jsonify({'retcode': -20001, 'message': 'Record not found', 'data': {}})
    except Exception as e:
        return jsonify({'retcode': -30000, 'message': 'Database error', 'data': {}})
    finally:
        if conn:
            conn.close()


@app.route('/open_kf_api/batch_delete_intervene_record', methods=['POST'])
@token_required
def batch_delete_intervene_record():
    data = request.json
    id_list = data.get('id_list')

    if not id_list or not isinstance(id_list, list) or len(id_list) == 0:
        return jsonify({'retcode': -20000, 'message': 'Missing or invalid mandatory parameter: id_list', 'data': {}})

    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # Retrieve the queries to delete their corresponding Cache entries
        cur.execute(f'SELECT query FROM t_user_qa_intervene_tab WHERE id IN ({",".join(["?"]*len(id_list))})', id_list)
        rows = cur.fetchall()

        for row in rows:
            query = row['query']
            key = f"open_kf:intervene:{query}"
            g_diskcache_client.delete(key)

        # Then, batch delete from DB
        try:
            with g_diskcache_lock.lock():
                cur.execute(f'DELETE FROM t_user_qa_intervene_tab WHERE id IN ({",".join(["?"]*len(id_list))})', id_list)
                conn.commit()
        except Exception as e:
            logger.error(f"process g_discache_lock exception:{e}")

        return jsonify({"retcode": 0, "message": "success", 'data': {}})
    except Exception as e:
        return jsonify({'retcode': -30000, 'message': 'Database error', 'data': {}})
    finally:
        if conn:
            conn.close()


@app.route('/open_kf_api/update_intervene_record', methods=['POST'])
@token_required
def update_intervene_record():
    data = request.json
    record_id = data.get('id')
    intervene_answer = data.get('intervene_answer')
    source = data.get('source', [])

    if None in (record_id, intervene_answer, source):
        return jsonify({'retcode': -20000, 'message': 'Missing or invalid mandatory parameters', 'data': {}})

    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        source_json = json.dumps(source)  # Convert the source list to a JSON string for storing in DB
        timestamp = int(time.time())
        # Update the DB record
        try:
            with g_diskcache_lock.lock():
                cur.execute('UPDATE t_user_qa_intervene_tab SET intervene_answer = ?, source = ?, mtime = ? WHERE id = ?',
                            (intervene_answer, source_json, timestamp, record_id))
                conn.commit()
        except Exception as e:
            logger.error(f"process g_discache_lock exception:{e}")

        # Retrieve the query text to update the corresponding Cache entry
        cur.execute('SELECT query FROM t_user_qa_intervene_tab WHERE id = ?', (record_id,))
        row = cur.fetchone()
        if row:
            query = row['query']
            key = f"open_kf:intervene:{query}"
            value = json.dumps({"answer": intervene_answer, "source": source})
            g_diskcache_client.set(key, value)
        else:
            return jsonify({'retcode': -20001, 'message': 'Record not found', 'data': {}})

        return jsonify({"retcode": 0, "message": "success", 'data': {}})
    except Exception as e:
        return jsonify({'retcode': -30000, 'message': 'Database error', 'data': {}})
    finally:
        if conn:
            conn.close()


@app.route('/open_kf_api/get_intervene_query_list', methods=['POST'])
@token_required
def get_intervene_query_list():
    data = request.json
    start_timestamp = data.get('start_timestamp')
    end_timestamp = data.get('end_timestamp')
    page = data.get('page')
    page_size = data.get('page_size')

    # Validate mandatory parameters
    if None in (start_timestamp, end_timestamp, page, page_size):
        return jsonify({'retcode': -20000, 'message': 'Missing mandatory parameters', 'data': {}})

    try:
        # Convert timestamps and pagination parameters to integers
        start_timestamp = int(start_timestamp)
        end_timestamp = int(end_timestamp)
        page = int(page)
        page_size = int(page_size)
    except ValueError:
        return jsonify({'retcode': -20001, 'message': 'Invalid parameters', 'data': {}})

    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # Calculate total count
        cur.execute('SELECT COUNT(*) FROM t_user_qa_intervene_tab WHERE ctime BETWEEN ? AND ?', (start_timestamp, end_timestamp))
        total_count = cur.fetchone()[0]

        # Calculate the starting point for the query
        start = (page - 1) * page_size

        # Retrieve the specified page of records
        cur.execute('''
            SELECT id, query, intervene_answer, source, ctime, mtime
            FROM t_user_qa_intervene_tab
            WHERE ctime BETWEEN ? AND ?
            ORDER BY ctime DESC
            LIMIT ? OFFSET ?''', (start_timestamp, end_timestamp, page_size, start))

        rows = cur.fetchall()
        record_list = [dict(row) for row in rows]  # Convert rows to dictionaries
        # Apply json.loads on the 'source' field of each record
        for record in record_list:
            if 'source' in record:  # Ensure the 'source' key exists
                try:
                    record['source'] = json.loads(record['source'])  # Convert JSON string to Python list
                except json.JSONDecodeError:
                    record['source'] = []  # If decoding fails, set to an empty list or other default value

        return jsonify({
            "retcode": 0,
            "message": "success",
            "data": {
                "total_count": total_count,
                "intervene_list": record_list
            }
        })
    except Exception as e:
        return jsonify({'retcode': -30000, 'message': 'Database error', 'data': {}})
    finally:
        if conn:
            conn.close()


@app.route('/open_kf_api/login', methods=['POST'])
def login():
    data = request.json
    account_name = data.get('account_name')
    password = data.get('password')

    if not account_name or not password:
        return jsonify({'retcode': -20000, 'message': 'Account name and password are required', 'data': {}})

    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # Check if the account exists and verify the password
        cur.execute('SELECT id, password_hash FROM t_account_tab WHERE account_name = ?', (account_name,))
        account = cur.fetchone()

        if account and check_password_hash(account['password_hash'], password):
            # Generate token with account_name in the payload
            token = TokenHelper.generate_token(account_name)
            logger.info(f"Generate token:'{token}'")
            
            # Set is_login to 1 and update mtime to the current Unix timestamp
            try:
                with g_diskcache_lock.lock():
                    cur.execute('UPDATE t_account_tab SET is_login = 1, mtime = ? WHERE account_name = ?', (int(time.time()), account_name,))
                    conn.commit()
            except Exception as e:
                logger.error(f"process g_discache_lock exception:{e}")

            return jsonify({'retcode': 0, 'message': 'Login successful', 'data': {'token': token}})
        else:
            return jsonify({'retcode': -20001, 'message': 'Invalid credentials', 'data': {}})
    except Exception as e:
        return jsonify({'retcode': -30000, 'message': f'An error occurred during login, exception:{e}', 'data': {}})
    finally:
        if conn:
            conn.close()


@app.route('/open_kf_api/update_password', methods=['POST'])
@token_required
def update_password():
    data = request.json
    account_name = data.get('account_name')
    current_password = data.get('current_password')
    new_password = data.get('new_password')

    if None in (account_name, current_password, new_password):
        return jsonify({'retcode': -20000, 'message': 'Account name, current password, and new password are required', 'data': {}})

    token_user_id = request.user_payload['user_id']
    if token_user_id != account_name:
        logger.error(f"account_name:'{account_name}' does not match with token_user_id:'{token_user_id}'")
        return jsonify({'retcode': -20001, 'message': 'Token is invalid!', 'data': {}})

    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # Check if the account exists and verify the current password
        cur.execute('SELECT id, password_hash FROM t_account_tab WHERE account_name = ?', (account_name,))
        account = cur.fetchone()

        if not account or not check_password_hash(account['password_hash'], current_password):
            logger.error(f"Invalid account_name:'{account_name}' or current_password:'{current_password}'")
            return jsonify({'retcode': -20001, 'message': 'Invalid account name or password', 'data': {}})

        # Update the password
        new_password_hash = generate_password_hash(new_password, method='pbkdf2:sha256', salt_length=10)
        try:
            with g_diskcache_lock.lock():
                cur.execute('UPDATE t_account_tab SET password_hash = ?, mtime = ? WHERE account_name = ?', (new_password_hash, int(time.time()), account_name,))
                conn.commit()
        except Exception as e:
            logger.error(f"process g_discache_lock exception:{e}")

        return jsonify({'retcode': 0, 'message': 'Password updated successfully', 'data': {}})
    except Exception as e:
        return jsonify({'retcode': -20001, 'message': f'An error occurred: {str(e)}', 'data': {}})
    finally:
        if conn:
            conn.close()


@app.route('/open_kf_api/get_bot_setting', methods=['POST'])
def get_bot_setting():
    """Retrieve bot setting, first trying Cache and falling back to DB if not found."""
    try:
        # Attempt to retrieve the setting from Cache
        key = "open_kf:bot_setting"
        setting_cache = g_diskcache_client.get(key)
        if setting_cache:
            setting_data = json.loads(setting_cache)
            return jsonify({'retcode': 0, 'message': 'Success', 'data': {'config': setting_data}})
        else:
            logger.warning(f"could not find '{key}' in Cache!")
    except Exception as e:
        logger.error(f"Error retrieving setting from Cache, excpetion:{e}")
        # Just ignore Cache error
        #return jsonify({'retcode': -30000, 'message': f'An error occurred: {str(e)}', 'data': {}})

    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute('SELECT * FROM t_bot_setting_tab LIMIT 1')
        setting = cur.fetchone()
        if setting:
            setting = dict(setting)
            # Process and return the setting details
            setting_data = {k: json.loads(v) if k in ['initial_messages', 'suggested_messages'] else v for k, v in setting.items()}

            # Add bot setting into Cache
            try:
                key = "open_kf:bot_setting"
                g_diskcache_client.set(key, json.dumps(setting_data))
            except Exception as e:
                logger.error(f" add bot setting into Cache is failed, the exception is {e}")
                # Just ignore Reids error

            return jsonify({'retcode': 0, 'message': 'Success', 'data': {'config': setting_data}})
        else:
            logger.warning(f"No setting found")
            return jsonify({'retcode': -20001, 'message': 'No setting found', 'data': {}})
    except Exception as e:
        logger.error(f"Error retrieving setting: {e}")
        return jsonify({'retcode': -30000, 'message': f'An error occurred: {str(e)}', 'data': {}})
    finally:
        if conn:
            conn.close()


@app.route('/open_kf_api/update_bot_setting', methods=['POST'])
@token_required
def update_bot_setting():
    data = request.json
    # Extract and validate all required fields
    setting_id = data.get('id')
    initial_messages = data.get('initial_messages')
    suggested_messages = data.get('suggested_messages')
    bot_name = data.get('bot_name')
    bot_avatar = data.get('bot_avatar')
    chat_icon = data.get('chat_icon')
    placeholder = data.get('placeholder')
    model = data.get('model')

    # Check for the presence of all required fields
    if None in (setting_id, initial_messages, suggested_messages, bot_name, bot_avatar, chat_icon, placeholder, model):
        return jsonify({'retcode': -20000, 'message': 'All fields are required', 'data': {}})

    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # Check if the setting with provided ID exists
        cur.execute('SELECT id FROM t_bot_setting_tab WHERE id = ?', (setting_id,))
        if not cur.fetchone():
            logger.error(f"No setting found")
            return jsonify({'retcode': -20001, 'message': 'Setting not found', 'data': {}})

        # Convert lists to JSON strings for storage
        initial_messages_json = json.dumps(initial_messages)
        suggested_messages_json = json.dumps(suggested_messages)

        # Update bot setting in DB
        timestamp = int(time.time())
        try:
            with g_diskcache_lock.lock():
                cur.execute('''
                    UPDATE t_bot_setting_tab
                    SET initial_messages = ?, suggested_messages = ?, bot_name = ?, bot_avatar = ?, chat_icon = ?, placeholder = ?, model = ?, mtime = ?
                    WHERE id = ?
                ''', (initial_messages_json, suggested_messages_json, bot_name, bot_avatar, chat_icon, placeholder, model, timestamp, setting_id))
                conn.commit()
        except Exception as e:
            logger.error(f"process g_discache_lock exception:{e}")

        # Update bot setting in Cache
        try:
            key = "open_kf:bot_setting"
            bot_setting = {
                'id': setting_id,
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
            g_diskcache_client.set(key, json.dumps(bot_setting))
        except Exception as e:
            logger.error(f"update bot seeting in Cache is failed, the exception is {e}")
            return jsonify({'retcode': -20001, 'message': f'An error occurred: {str(e)}', 'data': {}})

        return jsonify({'retcode': 0, 'message': 'Settings updated successfully', 'data': {}})
    except Exception as e:
        logger.error(f"Error updating setting in DB: {str(e)}")
        return jsonify({'retcode': -30000, 'message': f'An error occurred: {str(e)}', 'data': {}})
    finally:
        if conn:
            conn.close()


def is_valid_url(url):
    """Check if the provided string is a valid URL."""
    parsed_url = urlparse(url)
    return bool(parsed_url.scheme) and bool(parsed_url.netloc)

def async_crawl_link_task(site, version):
    """Start the crawl link task in an asyncio event loop."""
    logger.info(f"create crawler_link")
    crawler_link = AsyncCrawlerSiteLink(
        base_url=site,
        sqlite_db_path=f"{SQLITE_DB_DIR}/{SQLITE_DB_NAME}",
        max_requests=MAX_CRAWL_PARALLEL_REQUEST,
        version=version,
        distributed_lock=g_diskcache_lock
    )
    logger.info(f"async_crawl_link_task begin!, site:'{site}', version:{version}")
    asyncio.run(crawler_link.run())
    logger.info(f"async_crawl_link_task end!, site:'{site}', version:{version}")

@app.route('/open_kf_api/submit_crawl_site', methods=['POST'])
@token_required
def submit_crawl_site():
    """Submit a site for crawling."""
    data = request.json
    site = data.get('site')
    timestamp = data.get('timestamp')

    if not site or not timestamp:
        return jsonify({'retcode': -20000, 'message': 'site and timestamp are required', 'data': {}})

    if not is_valid_url(site):
        logger.error(f"site:'{site} is not a valid URL!")
        return jsonify({'retcode': -20001, 'message': f"site:'{site}' is not a valid URL", 'data': {}})

    domain = urlparse(site).netloc
    logger.info(f"domain is '{domain}'")
    conn = None
    try:
        timestamp = int(timestamp)
        conn = get_db_connection()
        cur = conn.cursor()

        # Check if the domain exists in the database
        cur.execute("SELECT id, version FROM t_domain_tab WHERE domain = ?", (domain,))
        domain_info = cur.fetchone()

        if domain_info and timestamp <= domain_info["version"]:
            return jsonify({'retcode': -20001, 'message': f'New timestamp:{timestamp} must be greater than the current version:{domain_info["version"]}.', 'data': {}})

        try:
            with g_diskcache_lock.lock():
                if domain_info:
                    domain_id, version = domain_info
                    # Update domain record
                    cur.execute("UPDATE t_domain_tab SET version = ?, domain_status = 1, mtime=? WHERE id = ?", (timestamp, int(time.time()), domain_id))
                else:
                    # Insert new domain record
                    cur.execute("INSERT INTO t_domain_tab (domain, domain_status, version, ctime, mtime) VALUES (?, 1, ?, ?, ?)", (domain, timestamp, int(time.time()), int(time.time())))

                conn.commit()
        except Exception as e:
            logger.error(f"process g_discache_lock exception:{e}")

        # Start the asynchronous crawl task
        Thread(target=async_crawl_link_task, args=(site, timestamp)).start()

        return jsonify({'retcode': 0, 'message': 'Site submitted successfully for crawling.', 'data': {}})
    except Exception as e:
        return jsonify({'retcode': -30000, 'message': f'An error occurred: {str(e)}', 'data': {}})
    finally:
        if conn:
            conn.close()


@app.route('/open_kf_api/get_crawl_site_info', methods=['POST'])
@token_required
def get_crawl_site_info():
    """Fetch the crawl status information for a given site or all sites if site parameter is not provided."""
    data = request.json
    site = data.get('site', None)  # Make site an optional parameter

    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        if site:
            if not is_valid_url(site):
                logger.error(f"site:'{site}' is not a valid URL!")
                return jsonify({'retcode': -20001, 'message': f"site:'{site}' is not a valid URL", 'data': {}})
            domain = urlparse(site).netloc
            logger.info(f"Searching for domain: '{domain}'")
            cur.execute("SELECT * FROM t_domain_tab WHERE domain = ?", (domain,))
        else:
            logger.info("Fetching information for all sites.")
            cur.execute("SELECT * FROM t_domain_tab")

        rows = cur.fetchall()
        if rows:
            sites_info = [dict(row) for row in rows]
            return jsonify({'retcode': 0, 'message': 'Success', 'data': {'sites_info': sites_info}})
        else:
            return jsonify({'retcode': -20001, 'message': 'No site information found', 'data': {}})
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return jsonify({'retcode': -30000, 'message': f'An error occurred: {str(e)}', 'data': {}})
    finally:
        if conn:
            conn.close()


@app.route('/open_kf_api/get_crawl_url_list', methods=['POST'])
@token_required
def get_crawl_url_list():
    """Fetch the list of URLs and their status information. If the site is specified and valid, returns information for that site. Returns an error if the site is specified but invalid."""
    data = request.json
    site = data.get('site', None)  # Make site an optional parameter

    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        response_data = {
            'url_list': []
        }

        if site is not None:
            if not is_valid_url(site):
                logger.error(f"Provided site: '{site}' is not a valid URL.")
                return jsonify({'retcode': -20001, 'message': f"Provided site: '{site}' is not a valid URL.", 'data': {}})

            domain = urlparse(site).netloc
            logger.info(f"Fetching URL list for domain: '{domain}'")
            cur.execute("SELECT domain_status FROM t_domain_tab WHERE domain = ?", (domain,))
            domain_status_row = cur.fetchone()
            if domain_status_row:
                response_data['domain_status'] = domain_status_row['domain_status']
            cur.execute("SELECT id, url, content_length, doc_status, version, ctime, mtime FROM t_raw_tab WHERE domain = ?", (domain,))
        else:
            logger.info("Fetching URL list for all domains.")
            cur.execute("SELECT id, url, content_length, doc_status, version, ctime, mtime FROM t_raw_tab")

        rows = cur.fetchall()
        response_data['url_list'] = [dict(row) for row in rows]
        return jsonify({'retcode': 0, 'message': 'Success', 'data': response_data})
    except Exception as e:
        logger.error(f"An error occurred while fetching URL list: {str(e)}")
        return jsonify({'retcode': -30000, 'message': f'An error occurred: {str(e)}', 'data': {}})
    finally:
        if conn:
            conn.close()


def async_crawl_content_task(domain_list, url_dict, task_type):
    """
    Starts the asynchronous crawl and embedding process for a list of document IDs.

    task_type:
      1 - add_content
      2 - delete_content
      3 - update_content
    """

    """Start the crawl content task in an asyncio event loop."""
    logger.info(f"async_crawl_content_task begin! domain_lsit:{domain_list}, url_dict:{url_dict}, task_type:{task_type}")
    crawler_content = AsyncCrawlerSiteContent(
        domain_list=domain_list,
        sqlite_db_path=f"{SQLITE_DB_DIR}/{SQLITE_DB_NAME}",
        max_requests=MAX_CRAWL_PARALLEL_REQUEST,
        max_chunk_length=MAX_CHUNK_LENGTH,
        document_embedder_obj=g_document_embedder,
        distributed_lock=g_diskcache_lock
    )

    # Run the crawler
    if task_type == 1:
        asyncio.run(crawler_content.add_content(url_dict))
    elif task_type == 2:
        asyncio.run(crawler_content.delete_content(url_dict))
    elif task_type == 3:
        asyncio.run(crawler_content.update_content(url_dict))
    logger.info(f"async_crawl_content_task end!, domain_list:{domain_list}', task_type:{task_type}")

def check_crawl_content_task(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        data = request.json
        id_list = data.get('id_list')

        if not id_list or not isinstance(id_list, list) or len(id_list) == 0:
            return jsonify({'retcode': -20000, 'message': 'Invalid or missing id_list parameter'})

        conn = None
        try:
            conn = get_db_connection()
            cur = conn.cursor()

            placeholders = ', '.join(['?'] * len(id_list))
            cur.execute(f"SELECT id, domain, url FROM t_raw_tab WHERE id IN ({placeholders})", id_list)
            rows = cur.fetchall()

            if len(rows) != len(id_list):
                missing_ids = set(id_list) - set(row[0] for row in rows)
                return jsonify({'retcode': -20001, 'message': f'The following ids do not exist: {missing_ids}', 'data': {}})

            url_dict = {row["id"]: row["url"] for row in rows}
            domain_list = list(set(row["domain"] for row in rows))
            logger.info(f"domain_list is {domain_list}")

            # Store domain and url_dict in request for further use
            request.domain_list = domain_list
            request.url_dict = url_dict

            # Check and update domain_status in t_domain_tab for all domains in domain_list
            timestamp = int(time.time())
            for domain in domain_list:
                cur.execute("SELECT domain_status FROM t_domain_tab WHERE domain = ?", (domain,))
                domain_info = cur.fetchone()
                if domain_info and domain_info["domain_status"] < 3:
                    try:
                        with g_diskcache_lock.lock():
                            cur.execute("UPDATE t_domain_tab SET domain_status = 3, mtime = ? WHERE domain = ?", (timestamp, domain))
                            conn.commit()
                            logger.info(f"Updated domain_status to 3 for domain: '{domain}'")
                    except Exception as e:
                        logger.error(f"process g_discache_lock exception:{e}")
        except Exception as e:
            return jsonify({'retcode': -30000, 'message': f'An error occurred: {str(e)}', 'data': {}})
        finally:
            if conn:
                conn.close()
        return f(*args, **kwargs)
    return decorated_function


@app.route('/open_kf_api/add_crawl_url_list', methods=['POST'])
@check_crawl_content_task
@token_required
def add_crawl_url_list():
    domain_list = request.domain_list
    url_dict = request.url_dict
    # Use threading to avoid blocking the Flask application
    Thread(target=async_crawl_content_task, args=(domain_list, url_dict, 1)).start()
    return jsonify({'retcode': 0, 'message': 'Started processing the URL list.', 'data': {}})


@app.route('/open_kf_api/delete_crawl_url_list', methods=['POST'])
@check_crawl_content_task
@token_required
def delete_crawl_url_list():
    domain_list = request.domain_list
    url_dict = request.url_dict
    # Use threading to avoid blocking the Flask application
    Thread(target=async_crawl_content_task, args=(domain_list, url_dict, 2)).start()
    return jsonify({'retcode': 0, 'message': 'Started deleting the URL list embeddings.'})


@app.route('/open_kf_api/update_crawl_url_list', methods=['POST'])
@check_crawl_content_task
@token_required
def update_crawl_url_list():
    domain_list = request.domain_list
    url_dict = request.url_dict
    # Use threading to avoid blocking the Flask application
    Thread(target=async_crawl_content_task, args=(domain_list, url_dict, 3)).start()
    return jsonify({'retcode': 0, 'message': 'Started updating the URL list embeddings.'})


@app.route('/open_kf_api/upload_picture', methods=['POST'])
@token_required
def upload_picture():
    picture_file = request.files.get('picture_file')
    if not picture_file:
        logger.error("Missing required parameters picture_file")
        return jsonify({'retcode': -20000, 'message': 'Missing required parameters picture_file', data:{}})

    try:
        original_filename = secure_filename(picture_file.filename)
        extension = os.path.splitext(original_filename)[1]
        new_filename = f"open_kf_{uuid.uuid4()}{extension}"

        save_directory = f"{STATIC_DIR}/{MEDIA_DIR}/"
        if not os.path.exists(save_directory):
            logger.error(f"save_directory:'{save_directory} does not exist!")
            return jsonify({'retcode': -20001, 'message': 'save_directory does not exitst!', 'data':{}})
        image_path = os.path.join(save_directory, new_filename)
        picture_file.save(image_path)

        picture_url  = f"{URL_PREFIX}{MEDIA_DIR}/{new_filename}"
        return jsonify({'retcode': 0, 'message': 'upload picture success', 'data': {'picture_url': picture_url}})
    except Exception as e:
        logger.error(f"An error occureed: {str(e)}")
        return jsonify({'retcode': -30000, 'message': f'An error occurred: {str(e)}', 'data': {}})


@app.route('/open_kf_api/get_url_sub_content_list', methods=['POST'])
@token_required
def get_url_sub_content_list():
    data = request.json
    url_id = data.get('id')
    if not url_id:
        return jsonify({'retcode': -20000, 'message': 'id is required', 'data': {}})

    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute('SELECT content FROM t_raw_tab WHERE id = ?', (url_id,))
        row = cur.fetchone()
        if row:
            content = row['content']
            content_vec = json.loads(content)
            sub_content_list = [
                {"index": index + 1, "content": part, "content_length": len(part)}
                for index, part in enumerate(content_vec)
            ]
            return jsonify({
                "retcode": 0,
                "message": "success",
                "data": {
                    "sub_content_list": sub_content_list
                }
            })
        else:
            return jsonify({'retcode': -30000, 'message': 'Content not found', 'data': {}})
    except Exception as e:
        return jsonify({'retcode': -30001, 'message': 'Database exception', 'data': {}})
    finally:
        if conn:
            conn.close()


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=7000)
