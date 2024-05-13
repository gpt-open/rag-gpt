from functools import wraps
import json
import os
from threading import Thread
import time
from typing import List, Dict, Any, Tuple
from urllib.parse import urlparse
from flask import Blueprint, Flask, request, Response
from langchain.schema.document import Document
from server.constant.constants import RECALL_TOP_K, RERANK_RECALL_TOP_K, MAX_QUERY_LENGTH, SESSION_EXPIRE_TIME, MAX_HISTORY_SESSION_LENGTH
from server.app.utils.decorators import token_required
from server.app.utils.sqlite_client import get_db_connection
from server.app.utils.diskcache_client import diskcache_client
from server.app.utils.diskcache_lock import diskcache_lock
from server.logger.logger_config import my_logger as logger
from server.rag.generation.cloud_llm import cloud_llm_generator
from server.rag.pre_retrieval.query_transformation.rewrite import query_rewrite
from server.rag.post_retrieval.rerank.flash_ranker import RerankRequest, reranker
from server.rag.retrieval.vector_search import vector_search


LLM_NAME = os.getenv('LLM_NAME')

MIN_RELEVANCE_SCORE = float(os.getenv('MIN_RELEVANCE_SCORE', '0.3'))
BOT_TOPIC = os.getenv('BOT_TOPIC')
USE_PREPROCESS_QUERY = int(os.getenv('USE_PREPROCESS_QUERY'))
USE_RERANKING = int(os.getenv('USE_RERANKING'))
USE_DEBUG = int(os.getenv('USE_DEBUG'))


queries_bp = Blueprint('queries', __name__, url_prefix='/open_kf_api/queries')


def get_user_query_history(user_id: str, is_streaming: bool) -> List[Any]:
    if is_streaming:
        history_key = f"open_kf:query_history:{user_id}:stream"
    else:
        history_key = f"open_kf:query_history:{user_id}"
    history_items = diskcache_client.get_list(history_key)[::-1]
    history = [json.loads(item) for item in history_items]
    return history


def save_user_query_history(user_id: str, query: str, answer: str, is_streaming: bool) -> None:
    try:
        # After generating the response from LLM
        # Store user query and LLM response in Cache
        if is_streaming:
            history_key = f"open_kf:query_history:{user_id}:stream:"
            history_data = {'query': query, 'answer': answer}
        else:
            history_key = f"open_kf:query_history:{user_id}"
            answer_json = json.loads(answer)
            history_data = {'query': query, 'answer': answer_json}
        diskcache_client.append_to_list(history_key, json.dumps(history_data), ttl=SESSION_EXPIRE_TIME, max_length=MAX_HISTORY_SESSION_LENGTH)
    except Exception as e:
        logger.error(f"For the query: '{query}' and user_id: '{user_id}', is processed failed with Cache, the exception is {e}")

    timestamp = int(time.time())
    conn = None
    try:
        # Store user query and LLM resposne in DB
        conn = get_db_connection()
        try:
            with diskcache_lock.lock():
                if is_streaming:
                    conn.execute('INSERT INTO t_user_qa_record_tab (user_id, query, answer, source, ctime, mtime) VALUES (?, ?, ?, ?, ?, ?)',
                             (user_id, query, answer, '[]', timestamp, timestamp))
                else:
                    conn.execute('INSERT INTO t_user_qa_record_tab (user_id, query, answer, source, ctime, mtime) VALUES (?, ?, ?, ?, ?, ?)',
                             (user_id, query, answer_json["answer"], json.dumps(answer_json["source"]), timestamp, timestamp))
                conn.commit()
        except Exception as e:
            logger.error(f"process discache_lock exception:{e}")
            return {'retcode': -30000, 'message': f'An error occurred: {e}', 'data': {}}
    except Exception as e:
        logger.error(f"For the query: '{query}' and user_id: '{user_id}', is processed failed with Database, the exception is {e}")
    finally:
        if conn:
            conn.close()


def preprocess_query(query: str, bot_topic: str) -> str:
    return query_rewrite(query, bot_topic)


def search_documents(query: str, k: int) -> List[Tuple[Document, float]]:
    beg_time = time.time()
    results = vector_search.similarity_search_with_relevance_scores(query, k)
    timecost = time.time() - beg_time
    logger.warning(f"search_documents, query: '{query}', k: {k}, the timecost is {timecost}")
    return results


def rerank_documents(query: str, results: List[Tuple[Document, float]]) -> List[Dict[str, Any]]:
    passages: List[Dict[str, Any]] = []
    index = 1
    for doc, chroma_score in results:
        item = {
            "id": index,
            "text": doc.page_content,
            "metadata": doc.metadata,
            "chroma_score": chroma_score
        }
        index += 1
        passages.append(item)

    beg_time = time.time()
    rerankrequest = RerankRequest(query=query, passages=passages)
    rerank_results = reranker.rerank(rerankrequest)
    timecost = time.time() - beg_time
    logger.warning(f"For the query: '{query}', rerank_documents, the timecost is {timecost}")

    if USE_DEBUG:
        rerank_info = "\n--------------------\n".join([
            f"ID: {item['id']}\nTEXT: {item['text']}\nMETADATA: {item['metadata']}\nCHROME_SCORE: {item['chroma_score']}\nSCORE: {item['score']}"
            for item in rerank_results
        ])
        logger.info(f"For the query: '{query}', the rerank results is:\n{rerank_info}")

    return rerank_results


def filter_documents(results: List[Tuple[Document, float]], min_relevance_score: float) -> Tuple[List[Tuple[Document, float]], set[str]]:
    filter_results = []
    domain_set = set()
    for doc, score in results:
        if score >= min_relevance_score:
            filter_results.append((doc, score))
            domain = urlparse(doc.metadata['source']).netloc
            domain_set.add(domain)
    return filter_results, domain_set


def filter_rerank_documents(rerank_results: List[Dict[str, Any]], min_relevance_score: float) -> Tuple[List[Dict[str, Any]], set[str]]:
    filter_rerank_results = []
    domain_set = set()
    for doc in rerank_results:
        if doc['chroma_score'] >= min_relevance_score:
            filter_rerank_results.append(doc)
            domain = urlparse(doc['metadata']['source']).netloc
            domain_set.add(domain)
    return filter_rerank_results, domain_set
    

def generate_answer(query: str, user_id: str, is_streaming: bool = False):
    bot_topic = BOT_TOPIC
    if USE_PREPROCESS_QUERY:
        adjust_query = preprocess_query(query, bot_topic)
    else:
        adjust_query = query

    if USE_RERANKING:
        top_k = RERANK_RECALL_TOP_K
    else:
        top_k = RECALL_TOP_K
    results = search_documents(adjust_query, top_k)
    if USE_DEBUG:
        results_info = "\n********************\n".join([
            f"URL: {doc.metadata['source']}\nscore: {score}\npage_content: {doc.page_content}"
            for doc, score in results
        ])
        logger.info(f"For the query: '{query}', user_id: '{user_id}', the recall results is\n{results_info}")

    filter_context = ''
    # Build the context with filtered documents, showing relevant documents
    if USE_RERANKING and results:
        # Rerank the documents
        rerank_results = rerank_documents(adjust_query, results)
        filter_rerank_results, recall_domain_set = filter_rerank_documents(rerank_results, MIN_RELEVANCE_SCORE)
        if filter_rerank_results:
            filter_context = "\n--------------------\n".join([
                f"URL: {doc['metadata']['source']}\nDocument: {doc['text']}"
                for doc in filter_rerank_results[:RECALL_TOP_K]
            ])
    else:
        filter_results, recall_domain_set = filter_documents(results, MIN_RELEVANCE_SCORE)
        if filter_results:
            filter_context = "\n--------------------\n".join([
                f"URL: {doc.metadata['source']}\nDocument: {doc.page_content}"
                for doc, score in filter_results
            ])

    context = ''
    if filter_context:
        # Get the history session from the cache
        history_session = get_user_query_history(user_id, is_streaming)
        if history_session:
            # Build the history context, showing user's historical queries and answers
            history_context = "\n--------------------\n".join([
                f"History Query: {item['query']}\nHistory Answer: {item['answer']}"
                for item in history_session
            ])
            context = f"""Documents (Sorted by Relevance Score from high to low):
{filter_context}

User Query History (Sorted by request time from most recent to oldest):
{history_context}"""
        else:
            context = f"""Documents (Sorted by Relevance Score from high to low):
{filter_context}"""
    else:
        # When no directly related documents are found, provide standard friendly response and guidance
        context = f"""No documents found directly related to the current query!
Please provide the response in the following format and ensure that the 'answer' part is translated into the same language as the user's query:

"Unfortunately, I cannot find a specific answer about '{query}' from the information provided. I'm here to assist you with information related to `{bot_topic}`. If you have any specific queries about our services or need help, feel free to ask, and I'll do my best to provide you with accurate and relevant answers."

Please ensure:
- Maintain the context and meaning of the original message.
- Translate the 'answer' to match the language of the user's query, enhancing user experience and understanding.
"""
    
    if not is_streaming:
        answer_format_prompt = """**Expected Response Format:**
The response should be a JSON object, with 'answer' and 'source' fields.
- "answer": "A detailed and specific answer, crafted in the query's language and fully formatted using **Markdown** syntax. **Don't repeat the `query`**",
- "source": ["List only unique URLs from the context that are directly related to the answer. Ensure that each URL is listed only once. If no documents are referenced, or the documents are not relevant, use an empty list []."]"""
    else:
        answer_format_prompt = """**Expected Response Format:**
The response should be fully formatted using **Mardown** syntax.
- A detailed and specific answer, crafted in the query's language. Don't start with 'Answer:' or 'answer:', just output the content. Don't repeat the `query`.
- Sources: ["List only unique URLs from the context that are directly related to the answer. Ensure that each URL is listed only once. If no documents are referenced, or the documents are not relevant, use an empty list []."]"""

    prompt = f"""
This smart customer service bot is designed to provide users with targeted information related to `{bot_topic}`.

If the query is similar to greeting phrases like ['Hi', 'Hello', 'Who are you?'], including greetings in other languages such as Chinese, Russian, French, Spanish, German, Japanese, Arabic, Hindi, Portuguese, Korean, etc. The bot will offer a friendly standard response, guiding users to seek information or services related to `{bot_topic}`.

Based on the documents and the user's historical queries, the bot aims to provide accurate and comprehensive answers in the language of the user's query. If the query is not related to any available documents, the user is informed that no relevant answer can be provided.

User ID: "{user_id}"
**Query:** "{query}"

**Context for Answering the Query:**
{context}

**Response Requirements:**
- If unsure about the answer, proactively seek clarification.
- Ensure that answers are strictly based on the provided context and are directly relevant to `{bot_topic}`.
- Inform users that queries unrelated to the provided context cannot be answered.
- Format the answer using Markdown syntax for clarity and readability.
- Craft responses in the same language as the query to enhance user understanding.

{answer_format_prompt}

Please format `answer` as follows:
The `answer` must be fully formatted using Markdown syntax to ensure proper rendering in the browser or APP. This includes:
- The `answer` must not be identical to the `query`.
- **Bold** (`**bold**`) and *italic* (`*italic*`) text for emphasis.
- Unordered lists (`- item`) for itemization and ordered lists (`1. item`) for sequencing.
- `Inline code` (`` `Inline code` ``) for brief code snippets and (` ``` `) for longer examples, specifying the programming language for syntax highlighting when possible.
- [Hyperlinks](URL) (`[Hyperlinks](URL)`) to reference external sources.
- Headings (`# Heading 1`, `## Heading 2`, ...) to structure the answer effectively.
"""

    if USE_DEBUG:
        logger.info(f"Prompt is:\n{prompt}")

    response = cloud_llm_generator.generate(prompt, is_streaming)
    return response, recall_domain_set


def check_smart_query(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        data = request.json
        user_id = data.get('user_id')
        query = data.get('query')
        if not user_id or not query:
            logger.error(f"user_id and query are required")
            return {'retcode': -20000, 'message': 'user_id and query are required', 'data': {}}, 400
        
        request.user_id = user_id
        request.query = query
        request.intervene_data = None

        try:
            # Check if the query is in Cache
            key = f"open_kf:intervene:{query}"
            intervene_data = diskcache_client.get(key)
            if intervene_data:
                logger.info(f"For the query: '{query}' and user_id: '{user_id}', is hit in Cache, the intervene_data is {intervene_data}")
                request.intervene_data = intervene_data
        except Exception as e:
            logger.error(f"Cache exception {e} for user_id: '{user_id}' and query: '{query}'")
        return f(*args, **kwargs)
    return decorated_function


def postprocess_llm_response(query: str, answer_json: Dict[str, Any], bot_topic: str, recall_domain_set: set[str]) -> bool:
    if answer_json['source']:
        answer_json['source'] = list(dict.fromkeys(answer_json['source']))

    is_adjusted = False
    adjust_source = []
    for url in answer_json['source']:
        domain = urlparse(url).netloc
        if domain in recall_domain_set:
            adjust_source.append(url)
        else:
            logger.warning(f"The domain of url: '{url}' is '{domain}', it is not in {recall_domain_set}, it should not be returned!")
            if not is_adjusted:
                is_adjusted = True

    if is_adjusted:
        answer_json['source'] = adjust_source
        logger.warning(f"adjust_source:{adjust_source}")

    if not adjust_source:
        if bot_topic not in answer_json['answer']:
            adjust_answer = f"Unfortunately, I cannot find a specific answer about '{query}' from the information provided. I'm here to assist you with information related to `{bot_topic}`. If you have any specific queries about our services or need help, feel free to ask, and I'll do my best to provide you with accurate and relevant answers."
            logger.warning(f"adjust_answer:'{adjust_answer}'")
            if not is_adjusted:
                is_adjusted = True
            answer_json['answer'] = adjust_answer
    return is_adjusted


@queries_bp.route('/smart_query', methods=['POST'])
@check_smart_query
@token_required
def smart_query():
    try:
        user_id = request.user_id
        query = request.query
        intervene_data = request.intervene_data
        if intervene_data:
            # Start a new thread to execute saving history records asynchronously
            Thread(target=save_user_query_history, args=(user_id, query, intervene_data, False)).start()
            intervene_data_json = json.loads(intervene_data)
            return {"retcode": 0, "message": "success", "data": intervene_data_json}

        if len(query) > MAX_QUERY_LENGTH:
            query = query[:MAX_QUERY_LENGTH]

        beg_time = time.time()
        response, recall_domain_set = generate_answer(query, user_id, False)
        logger.warning(f"[Track token consumption] for query: '{query}', usage={response.usage}")
        answer = response.choices[0].message.content

        #logger.warning(f"The answer is:\n{answer}")
        if LLM_NAME == 'ZhipuAI':
            #logger.warning(f"The answer is:\n{answer}")

            # Solve the result format problem of ZhipuAI
            if answer.startswith("```json"):
                answer = answer[7:]
                if answer.endswith("```"):
                    answer = answer[:-3]

        timecost = time.time() - beg_time
        answer_json = json.loads(answer)
        answer_json["source"] = list(dict.fromkeys(answer_json["source"]))
        logger.success(f"For smart_query, query: '{query}' and user_id: '{user_id}', is processed successfully, the answer is:\n{answer}\nthe total timecost is {timecost}\n")

        #is_adjusted = postprocess_llm_response(query, answer_json, BOT_TOPIC, recall_domain_set)
        #if is_adjusted:
        #    answer = json.dumps(answer_json)

        # Start another new thread to execute saving history records asynchronously
        Thread(target=save_user_query_history, args=(user_id, query, answer, False)).start()
        return {"retcode": 0, "message": "success", "data": answer_json}
    except Exception as e:
        logger.error(f"For the query: '{query}' and user_id: '{user_id}', is processed failed, the exception is {e}")
        return {'retcode': -20001, 'message': str(e), 'data': {}}


@queries_bp.route('/smart_query_stream', methods=['POST'])
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
            save_user_query_history(user_id, query, intervene_data, True)

            def generate_intervene():
                yield intervene_data
            return Response(generate_intervene(), mimetype="text/event-stream", headers=headers)

        if len(query) > MAX_QUERY_LENGTH:
            query = query[:MAX_QUERY_LENGTH]

        beg_time = time.time()
        def generate_llm():
            answer_chunks = []
            response, _ = generate_answer(query, user_id, True)
            for chunk in response:
                logger.info(f"chunk is: {chunk}")
                content = chunk.choices[0].delta.content
                if content:
                    answer_chunks.append(content)
                    # Send each answer segment
                    yield content

                if LLM_NAME == 'ZhipuAI':
                    if chunk.usage:
                        logger.warning(f"[Track token consumption of streaming] for query: '{query}', usage={chunk.usage}")
            # After the streaming response is complete, save to Cache and SQLite
            answer = ''.join(answer_chunks)
            timecost = time.time() - beg_time
            logger.success(f"query: '{query}' and user_id: '{user_id}' is processed successfully, the answer is:\n{answer}\nthe total timecost is {timecost}\n")
            save_user_query_history(user_id, query, answer, True)
        return Response(generate_llm(), mimetype="text/event-stream", headers=headers)
    except Exception as e:
        logger.error(f"query: '{query}' and user_id: '{user_id}' is processed failed, the exception is {e}")
        return {'retcode': -30000, 'message': str(e), 'data': {}}


@queries_bp.route('/get_user_conversation_list', methods=['POST'])
@token_required
def get_user_conversation_list():
    """Retrieve a list of user conversations within a specified time range, with pagination and total count."""
    data = request.json
    start_timestamp = data.get('start_timestamp')
    end_timestamp = data.get('end_timestamp')
    page = data.get('page')
    page_size = data.get('page_size')

    if None in ([start_timestamp, end_timestamp, page, page_size]):
        return {'retcode': -20000, 'message': 'Missing required parameters'}

    if not isinstance(start_timestamp, int) or not isinstance(end_timestamp, int):
        return {'retcode': -20001, 'message': 'Invalid start_timestamp or end_timestamp parameters', 'data': {}}

    if not isinstance(page, int) or not isinstance(page_size, int) or page < 1 or page_size < 1:
        return {'retcode': -20001, 'message': 'Invalid page or page_size parameters', 'data': {}}

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

        return {'retcode': 0, 'message': 'Success', 'data': {'total_count': total_count, 'conversation_list': conversation_list}}
    except Exception as e:
        logger.error(f"Failed to retrieve user conversation list: {e}")
        return {'retcode': -30000, 'message': 'Internal server error'}
    finally:
        if conn:
            conn.close()


@queries_bp.route('/get_user_query_history_list', methods=['POST'])
@token_required
def get_user_query_history_list():
    data = request.json
    page = data.get('page')
    page_size = data.get('page_size')
    user_id = data.get('user_id')

    # Check for mandatory parameters
    if None in (page, page_size, user_id):
        logger.error("page, page_size and user_id are required")
        return {'retcode': -20000, 'message': 'page, page_size and user_id are required', 'data': {}}

    try:
        # Convert timestamps and pagination parameters to integers
        page = int(page)
        page_size = int(page_size)
    except ValueError as e:
        logger.error(f"Parameter conversion error: {e}")
        return {'retcode': -20001, 'message': 'Invalid parameters', 'data': {}}

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

        return {
            "retcode": 0,
            "message": "success",
            "data": {
                "total_count": total_count,
                "query_list": record_list
            }
        }
    except Exception as e:
        logger.error(f"Database exception: {e}")
        return {'retcode': -30000, 'message': 'Database exception', 'data': {}}
    finally:
        if conn:
            conn.close()
