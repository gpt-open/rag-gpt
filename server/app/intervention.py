import json
import time
from flask import Blueprint, request
from server.app.utils.decorators import token_required
from server.app.utils.sqlite_client import get_db_connection
from server.app.utils.diskcache_client import diskcache_client
from server.app.utils.diskcache_lock import diskcache_lock
from server.logger.logger_config import my_logger as logger

intervention_bp = Blueprint('intervention',
                            __name__,
                            url_prefix='/open_kf_api/intervention')


@intervention_bp.route('/add_intervene_record', methods=['POST'])
@token_required
def add_intervene_record():
    data = request.json
    query = data.get('query')
    intervene_answer = data.get('intervene_answer')
    source = data.get('source', [])

    if None in (query, intervene_answer, source):
        return {
            'retcode': -20000,
            'message': 'Missing mandatory parameters',
            'data': {}
        }

    conn = None
    try:
        # Check if query already exists in the database
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute(
            'SELECT COUNT(*) FROM t_user_qa_intervene_tab WHERE query = ?',
            (query, ))
        result = cur.fetchone()
        if result and result[0] > 0:
            logger.error(
                f"intervene query:'{query}' is already exists in the database")
            return {
                'retcode': -30000,
                'message': 'Query already exists in the database',
                'data': {}
            }

        # Insert the intervene record into DB
        timestamp = int(time.time())
        source_str = json.dumps(source)

        try:
            with diskcache_lock.lock():
                cur.execute(
                    'INSERT INTO t_user_qa_intervene_tab (query, intervene_answer, source, ctime, mtime) VALUES (?, ?, ?, ?, ?)',
                    (query, intervene_answer, source_str, timestamp,
                     timestamp))
                conn.commit()
        except Exception as e:
            logger.error(f"process discache_lock exception:{e}")
            return {
                'retcode': -30000,
                'message': f'An error occurred: {e}',
                'data': {}
            }

        # Update Cache using simple string with the query as the key (prefixed)
        key = f"open_kf:intervene:{query}"
        value = json.dumps({"answer": intervene_answer, "source": source})
        diskcache_client.set(key, value)

        return {"retcode": 0, "message": "success", 'data': {}}
    except Exception as e:
        return {
            'retcode': -30000,
            'message': 'Database or Cache error',
            'data': {}
        }
    finally:
        if conn:
            conn.close()


@intervention_bp.route('/delete_intervene_record', methods=['POST'])
@token_required
def delete_intervene_record():
    data = request.json
    record_id = data.get('id')

    if not record_id:
        return {'retcode': -20000, 'message': 'id is required', 'data': {}}

    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # First, find the query string for the given id to delete it from Cache
        cur.execute('SELECT query FROM t_user_qa_intervene_tab WHERE id = ?',
                    (record_id, ))
        row = cur.fetchone()

        if row:
            query = row['query']
            # Delete the record from DB
            try:
                with diskcache_lock.lock():
                    cur.execute(
                        'DELETE FROM t_user_qa_intervene_tab WHERE id = ?',
                        (record_id, ))
                    conn.commit()
            except Exception as e:
                logger.error(f"process g_discache_lock exception:{e}")
                return {
                    'retcode': -30000,
                    'message': f'An error occurred: {e}',
                    'data': {}
                }

            # Now, delete the corresponding record from Cache
            key = f"open_kf:intervene:{query}"
            diskcache_client.delete(key)

            return {"retcode": 0, "message": "success", 'data': {}}
        else:
            return {
                'retcode': -20001,
                'message': 'Record not found',
                'data': {}
            }
    except Exception as e:
        return {'retcode': -30000, 'message': 'Database error', 'data': {}}
    finally:
        if conn:
            conn.close()


@intervention_bp.route('/batch_delete_intervene_record', methods=['POST'])
@token_required
def batch_delete_intervene_record():
    data = request.json
    id_list = data.get('id_list')

    if not id_list or not isinstance(id_list, list) or len(id_list) == 0:
        return {
            'retcode': -20000,
            'message': 'Missing or invalid mandatory parameter: id_list',
            'data': {}
        }

    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # Retrieve the queries to delete their corresponding Cache entries
        cur.execute(
            f'SELECT query FROM t_user_qa_intervene_tab WHERE id IN ({",".join(["?"]*len(id_list))})',
            id_list)
        rows = cur.fetchall()

        for row in rows:
            query = row['query']
            key = f"open_kf:intervene:{query}"
            diskcache_client.delete(key)

        # Then, batch delete from DB
        try:
            with diskcache_lock.lock():
                cur.execute(
                    f'DELETE FROM t_user_qa_intervene_tab WHERE id IN ({",".join(["?"]*len(id_list))})',
                    id_list)
                conn.commit()
        except Exception as e:
            logger.error(f"process discache_lock exception:{e}")
            return {
                'retcode': -30000,
                'message': f'An error occurred: {e}',
                'data': {}
            }

        return {"retcode": 0, "message": "success", 'data': {}}
    except Exception as e:
        return {'retcode': -30000, 'message': 'Database error', 'data': {}}
    finally:
        if conn:
            conn.close()


@intervention_bp.route('/update_intervene_record', methods=['POST'])
@token_required
def update_intervene_record():
    data = request.json
    record_id = data.get('id')
    intervene_answer = data.get('intervene_answer')
    source = data.get('source', [])

    if None in (record_id, intervene_answer, source):
        return {
            'retcode': -20000,
            'message': 'Missing or invalid mandatory parameters',
            'data': {}
        }

    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # Convert the source list to a JSON string for storing in DB
        source_json = json.dumps(source)
        timestamp = int(time.time())
        # Update the DB record
        try:
            with diskcache_lock.lock():
                cur.execute(
                    'UPDATE t_user_qa_intervene_tab SET intervene_answer = ?, source = ?, mtime = ? WHERE id = ?',
                    (intervene_answer, source_json, timestamp, record_id))
                conn.commit()
        except Exception as e:
            logger.error(f"process discache_lock exception:{e}")
            return {
                'retcode': -30000,
                'message': f'An error occurred: {e}',
                'data': {}
            }

        # Retrieve the query text to update the corresponding Cache entry
        cur.execute('SELECT query FROM t_user_qa_intervene_tab WHERE id = ?',
                    (record_id, ))
        row = cur.fetchone()
        if row:
            query = row['query']
            key = f"open_kf:intervene:{query}"
            value = json.dumps({"answer": intervene_answer, "source": source})
            diskcache_client.set(key, value)
        else:
            return {
                'retcode': -20001,
                'message': 'Record not found',
                'data': {}
            }

        return {"retcode": 0, "message": "success", 'data': {}}
    except Exception as e:
        return {'retcode': -30000, 'message': 'Database error', 'data': {}}
    finally:
        if conn:
            conn.close()


@intervention_bp.route('/get_intervene_query_list', methods=['POST'])
@token_required
def get_intervene_query_list():
    data = request.json
    start_timestamp = data.get('start_timestamp')
    end_timestamp = data.get('end_timestamp')
    page = data.get('page')
    page_size = data.get('page_size')

    # Validate mandatory parameters
    if None in (start_timestamp, end_timestamp, page, page_size):
        return {
            'retcode': -20000,
            'message': 'Missing mandatory parameters',
            'data': {}
        }

    if not isinstance(start_timestamp, int) or not isinstance(
            end_timestamp, int):
        return {
            'retcode': -20001,
            'message': 'Invalid start_timestamp or end_timestamp parameters',
            'data': {}
        }

    if not isinstance(page, int) or not isinstance(
            page_size, int) or page < 1 or page_size < 1:
        return {
            'retcode': -20001,
            'message': 'Invalid page or page_size parameters',
            'data': {}
        }

    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # Calculate total count
        cur.execute(
            'SELECT COUNT(*) FROM t_user_qa_intervene_tab WHERE ctime BETWEEN ? AND ?',
            (start_timestamp, end_timestamp))
        total_count = cur.fetchone()[0]

        # Calculate the starting point for the query
        start = (page - 1) * page_size

        # Retrieve the specified page of records
        cur.execute(
            '''
            SELECT id, query, intervene_answer, source, ctime, mtime
            FROM t_user_qa_intervene_tab
            WHERE ctime BETWEEN ? AND ?
            ORDER BY ctime DESC
            LIMIT ? OFFSET ?''',
            (start_timestamp, end_timestamp, page_size, start))

        rows = cur.fetchall()
        # Convert rows to dictionaries
        record_list = [dict(row) for row in rows]
        # Apply json.loads on the 'source' field of each record
        for record in record_list:
            if 'source' in record:  # Ensure the 'source' key exists
                try:
                    # Convert JSON string to Python list
                    record['source'] = json.loads(record['source'])
                except json.JSONDecodeError:
                    # If decoding fails, set to an empty list or other default value
                    record['source'] = []

        return {
            "retcode": 0,
            "message": "success",
            "data": {
                "total_count": total_count,
                "intervene_list": record_list
            }
        }
    except Exception as e:
        return {'retcode': -30000, 'message': 'Database error', 'data': {}}
    finally:
        if conn:
            conn.close()
