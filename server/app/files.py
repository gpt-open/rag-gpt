import aiofiles
import asyncio
from datetime import datetime
import json
import os
from threading import Thread
import time
from typing import Dict, List, Any
import uuid
from flask import Blueprint, request
from server.constant.constants import (MAX_LOCAL_FILE_BATCH_LENGTH,
                                       MAX_FILE_SIZE, LOCAL_FILE_DOWNLOAD_DIR,
                                       STATIC_DIR, FILE_LOADER_EXTENSIONS,
                                       MAX_CONCURRENT_WRITES,
                                       LOCAL_FILE_PROCESS_FAILED)
from server.app.utils.decorators import token_required
from server.app.utils.sqlite_client import get_db_connection
from server.app.utils.diskcache_lock import diskcache_lock
from server.app.utils.hash import generate_md5
from server.logger.logger_config import my_logger as logger
from server.rag.index.parser.file_loader.csv_loader import AsyncCsvLoader
from server.rag.index.parser.file_loader.docx_loader import AsyncDocxLoader
from server.rag.index.parser.file_loader.epub_loader import AsyncEpubLoader
from server.rag.index.parser.file_loader.html_loader import AsyncHtmlLoader
from server.rag.index.parser.file_loader.md_loader import AsyncMdLoader
from server.rag.index.parser.file_loader.mobi_loader import AsyncMobiLoader
from server.rag.index.parser.file_loader.pdf_loader import AsyncPdfLoader
from server.rag.index.parser.file_loader.pptx_loader import AsyncPptxLoader
from server.rag.index.parser.file_loader.txt_loader import AsyncTxtLoader
from server.rag.index.parser.file_loader.xlsx_loader import AsyncXlsxLoader
from server.rag.index.parser.file_parser.markdown_parser import AsyncTextParser

URL_PREFIX = os.getenv('URL_PREFIX')

files_bp = Blueprint('files', __name__, url_prefix='/open_kf_api/files')


async def write_file_async(file_path: str, content: bytes) -> None:
    """Asynchronously write content to a file."""
    async with aiofiles.open(file_path, 'wb') as f:
        await f.write(content)


async def parse_file_content_async(file_path: str, file_extension: str,
                                   file_md5: str,
                                   id_url_info: Dict[str, Any]) -> None:
    file_loader_obj = None
    if file_extension == ".csv":
        file_loader_obj = AsyncCsvLoader(file_path=file_path)
    elif file_extension == ".docx":
        file_loader_obj = AsyncDocxLoader(file_path=file_path)
    elif file_extension == ".epub":
        file_loader_obj = AsyncEpubLoader(file_path=file_path)
    elif file_extension == ".html":
        file_loader_obj = AsyncHtmlLoader(file_path=file_path)
    elif file_extension == ".md":
        file_loader_obj = AsyncMdLoader(file_path=file_path)
    elif file_extension == ".mobi":
        file_loader_obj = AsyncMobiLoader(file_path=file_path)
    elif file_extension == ".pdf":
        file_loader_obj = AsyncPdfLoader(file_path=file_path)
    elif file_extension == ".pptx":
        file_loader_obj = AsyncPptxLoader(file_path=file_path)
    elif file_extension == ".txt":
        file_loader_obj = AsyncTxtLoader(file_path=file_path)
    elif file_extension == ".xlsx":
        file_loader_obj = AsyncXlsxLoader(file_path=file_path)

    if file_loader_obj:
        content = await file_loader_obj.get_content()
        doc_id = id_url_info[file_md5]["id"]
        url = id_url_info[file_md5]["url"]
        text_parser_obj = AsyncTextParser()
        if content:
            await text_parser_obj.add_content(doc_id=doc_id,
                                              content=content,
                                              url=url)
        else:
            # if os.path.exists(file_path):
            #    os.remove(file_path)

            conn = None
            try:
                conn = get_db_connection()
                cur = conn.cursor()
                try:
                    with diskcache_lock.lock():
                        cur.execute(
                            'UPDATE t_local_file_tab SET doc_status = ?, content_md5 = '
                            ' WHERE id = ?',
                            (LOCAL_FILE_PROCESS_FAILED, doc_id))
                        conn.commit()
                except Exception as e:
                    logger.error(f"Process discache_lock exception: {e}")
            except Exception as e:
                logger.error(f"An error occurred: {e}")
            finally:
                if conn:
                    conn.close()
    else:
        logger.error(f"file_extension: '{file_extension}' is illegal!")


async def add_files_limited_by_semaphore(
        file_data: List[Dict[str, str]],
        id_url_info: Dict[str, Any],
        max_concurrent_writes: int = 5) -> None:
    """Save files with limited concurrency using a semaphore."""
    semaphore = asyncio.Semaphore(max_concurrent_writes)

    async def semaphore_write(data, id_url_info):
        async with semaphore:
            await write_file_async(data['file_path'], data['content'])

    # Create the event loop in the thread where this function is called
    loop = asyncio.get_event_loop()

    # Create and schedule coroutine tasks
    tasks = [
        loop.create_task(semaphore_write(data, id_url_info))
        for data in file_data
    ]
    await asyncio.gather(*tasks)

    for data in file_data:
        await parse_file_content_async(data['file_path'],
                                       data['file_extension'],
                                       data['file_md5'], id_url_info)


def add_local_file_info(file_data: List[Dict[str, str]],
                        id_url_info: Dict[str, Any],
                        max_concurrent_writes: int = 5):
    """Synchronously calls the asynchronous save function to write files with limited concurrency."""
    logger.info(
        f"[DOWNLOAD FILE] add_local_file_info begin, id_url_info: {id_url_info}"
    )
    beg_time = int(time.time())
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(
        add_files_limited_by_semaphore(file_data, id_url_info,
                                       max_concurrent_writes))
    loop.close()
    timecost = int(time.time()) - beg_time
    logger.warning(
        f"[DOWNLOAD FILE] add_local_file_info end, timecost is {timecost}")


@files_bp.route('/submit_local_file_list', methods=['POST'])
@token_required
def submit_local_file_list():
    if 'file_list' not in request.files:
        return {'retcode': -20000, 'message': 'file_list is required'}

    file_list = request.files.getlist('file_list')
    if len(file_list) > MAX_LOCAL_FILE_BATCH_LENGTH:
        logger.error(
            f'Too many files uploaded, the maximum is {MAX_LOCAL_FILE_BATCH_LENGTH}!'
        )
        return {
            'retcode':
            -20001,
            'message':
            f'Too many files uploaded, the maximum is {MAX_LOCAL_FILE_BATCH_LENGTH}!'
        }

    file_data = []
    md5_set = set()

    conn = None
    try:
        for file_ in file_list:
            file_.seek(0, os.SEEK_END)
            file_size = file_.tell()
            file_.seek(0)
            if file_size > MAX_FILE_SIZE:
                logger.error(
                    f'File {file_.filename} exceeds the size limit of {MAX_FILE_SIZE} bytes!'
                )
                return {
                    'retcode':
                    -20002,
                    'message':
                    f'File {file_.filename} exceeds the size limit of {MAX_FILE_SIZE} bytes!'
                }

            if file_size == 0:
                logger.error(f'File {file_.filename} is empty!')
                return {
                    'retcode': -20002,
                    'message': f'File {file_.filename} is empty!'
                }

            _, file_extension = os.path.splitext(file_.filename)
            if file_extension.lower() not in FILE_LOADER_EXTENSIONS:
                logger.error(
                    f"Unsupported file extension '{file_extension}' for {file_.filename}"
                )
                return {
                    'retcode':
                    -20003,
                    'message':
                    f"Unsupported file extension '{file_extension}' for {file_.filename}"
                }

            file_content = file_.read()
            file_md5 = generate_md5(file_content)

            if file_md5 in md5_set:
                logger.error(
                    f'Local duplicate file detected: {file_.filename}')
                return {
                    'retcode': -20004,
                    'message':
                    f'Local duplicate file detected: {file_.filename}'
                }
            md5_set.add(file_md5)

            day_folder = datetime.now().strftime("%Y_%m_%d")
            unique_folder = str(uuid.uuid4())
            save_directory = os.path.join(STATIC_DIR, LOCAL_FILE_DOWNLOAD_DIR,
                                          day_folder, unique_folder)
            os.makedirs(save_directory, exist_ok=True)

            file_path = os.path.join(save_directory, file_.filename)
            file_url = f"{URL_PREFIX}{STATIC_DIR}/{LOCAL_FILE_DOWNLOAD_DIR}/{day_folder}/{unique_folder}/{file_.filename}"

            file_data.append({
                'filename': file_.filename,
                'file_extension': file_extension.lower(),
                'content': file_content,
                'file_md5': file_md5,
                'file_path': file_path,
                'file_url': file_url,
                'file_size': file_size
            })

        conn = get_db_connection()
        cur = conn.cursor()
        placeholders = ', '.join(['?'] * len(md5_set))
        cur.execute(
            f"SELECT content_md5 FROM t_local_file_tab WHERE doc_status = 4 and content_md5 IN ({placeholders})",
            tuple(md5_set))
        existing_md5 = {row[0] for row in cur.fetchall()}

        duplicate_files = [
            data['filename'] for data in file_data
            if data['file_md5'] in existing_md5
        ]
        if duplicate_files:
            logger.error(
                f'Duplicate files found: {", ".join(duplicate_files)}')
            return {
                'retcode': -20005,
                'message':
                f'Duplicate files found: {", ".join(duplicate_files)}'
            }

        insert_data = []
        timestamp = int(time.time())
        for data in file_data:
            insert_data.append(
                (data['file_url'], data['filename'], data['file_path'],
                 data['file_extension'], data['file_size'], data['file_md5'],
                 1, timestamp, timestamp))

        try:
            with diskcache_lock.lock():
                cur.executemany(
                    '''
                    INSERT INTO t_local_file_tab (url, origin_file_name, file_path, file_type, content_length, content_md5, doc_status, ctime, mtime)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', insert_data)
                conn.commit()
        except Exception as e:
            logger.error(f"Process discache_lock exception: {e}")
            return {
                'retcode': -30000,
                'message': f'An error occurred: {e}',
                'data': {}
            }

        cur.execute(
            f"SELECT id, url, content_md5 FROM t_local_file_tab WHERE content_md5 IN ({placeholders})",
            tuple(md5_set))
        rows = cur.fetchall()
        inserted_ids = [row["id"] for row in rows]

        id_url_info = {}
        for row in rows:
            id_url_info[row["content_md5"]] = {
                "id": row["id"],
                "url": row["url"]
            }

        Thread(target=add_local_file_info,
               args=(file_data, id_url_info, MAX_CONCURRENT_WRITES)).start()

        return {
            'retcode': 0,
            'message': 'Files uploaded and metadata saved successfully',
            'data': {
                'file_id_list': inserted_ids
            }
        }
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return {
            'retcode': -30000,
            'message': f'An error occurred: {e}',
            'data': {}
        }
    finally:
        if conn:
            conn.close()


@files_bp.route('/get_local_file_list', methods=['POST'])
@token_required
def get_local_file_list():
    data = request.json
    file_id_list = data.get('id_list', None)  # Make site an optional parameter

    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        if file_id_list:
            placeholders = ', '.join(['?'] * len(file_id_list))
            cur.execute(
                f"SELECT id, url, origin_file_name, file_type, content_length, doc_status, ctime, mtime FROM t_local_file_tab WHERE id IN ({placeholders})",
                file_id_list)
        else:
            cur.execute(
                "SELECT id, url, origin_file_name, file_type, content_length, doc_status, ctime, mtime FROM t_local_file_tab"
            )

        rows = cur.fetchall()
        response_data = {}
        response_data['file_list'] = [dict(row) for row in rows]
        return {'retcode': 0, 'message': 'Success', 'data': response_data}
    except Exception as e:
        logger.error(f"An error occurred while fetching URL list: {e}")
        return {
            'retcode': -30000,
            'message': f'An error occurred: {e}',
            'data': {}
        }
    finally:
        if conn:
            conn.close()


async def delete_local_file_info_async(file_dict: Dict[int, str]) -> None:
    text_parser = AsyncTextParser()
    for doc_id in file_dict:
        await text_parser.delete_content(doc_id)

        file_path = file_dict[doc_id]
        if os.path.exists(file_path):
            os.remove(file_path)


def delete_local_file_info(file_dict: Dict[int, str]) -> None:
    logger.info(
        f"[DOWNLOAD FILE] delete_local_file_info beg, file_dict: {file_dict}")
    beg_time = int(time.time())
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(delete_local_file_info_async(file_dict))
    loop.close()
    timecost = int(time.time()) - beg_time
    logger.warning(
        f"[DOWNLOAD FILE] delete_local_file_info end, timecost is {timecost}")


@files_bp.route('/delete_local_file_list', methods=['POST'])
@token_required
def delete_local_file_list():
    data = request.json
    file_id_list = data.get('id_list')

    if not file_id_list:
        return {
            'retcode': -20000,
            'message': 'id_list is required',
            'data': {}
        }

    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        placeholders = ', '.join(['?'] * len(file_id_list))
        cur.execute(
            f"SELECT id, file_path FROM t_local_file_tab WHERE id IN ({placeholders})",
            file_id_list)
        file_dict = {row['id']: row['file_path'] for row in cur.fetchall()}

        # Use threading to avoid blocking the Flask application
        Thread(target=delete_local_file_info, args=(file_dict, )).start()

        return {
            'retcode': 0,
            'message': 'Started deleting the local file list embeddings.',
            'data': {}
        }
    except Exception as e:
        logger.error(f"An error occurred while deleting local file list: {e}")
        return {
            'retcode': -30000,
            'message': f'An error occurred: {e}',
            'data': {}
        }
    finally:
        if conn:
            conn.close()


@files_bp.route('/get_local_file_sub_content_list', methods=['POST'])
@token_required
def get_local_file_sub_content_list():
    data = request.json
    file_id = data.get('id')
    page = data.get('page')
    page_size = data.get('page_size')

    # Validate mandatory parameters
    if None in (file_id, page, page_size):
        return {
            'retcode': -20000,
            'message': 'Missing mandatory parameters',
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
            'SELECT COUNT(*) FROM t_local_file_chunk_tab WHERE file_id = ?',
            (file_id, ))
        total_count = cur.fetchone()[0]

        # Calculate the starting point for the query
        start = (page - 1) * page_size

        # Retrieve the specified page of records
        cur.execute(
            '''
            SELECT chunk_index as "index", content, content_length
            FROM t_local_file_chunk_tab
            WHERE file_id = ?
            ORDER BY chunk_index
            LIMIT ? OFFSET ?''', (file_id, page_size, start))

        rows = cur.fetchall()
        # Convert rows to dictionaries
        record_list = [dict(row) for row in rows]

        return {
            "retcode": 0,
            "message": "success",
            "data": {
                "total_count": total_count,
                "sub_content_list": record_list
            }
        }
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return {'retcode': -30001, 'message': 'Database exception', 'data': {}}
    finally:
        if conn:
            conn.close()
