import aiosqlite
import json
import time
from typing import List
from server.app.utils.diskcache_lock import diskcache_lock
from server.logger.logger_config import my_logger as logger
from server.constant.constants import (SQLITE_DB_DIR, SQLITE_DB_NAME,
                                       MAX_CHUNK_LENGTH, CHUNK_OVERLAP,
                                       FROM_LOCAL_FILE, LOCAL_FILE_PARSING,
                                       LOCAL_FILE_PARSING_COMPLETED,
                                       LOCAL_FILE_EMBEDDED,
                                       LOCAL_FILE_PROCESS_FAILED)
from server.rag.index.chunk.markdown_splitter import MarkdownTextSplitter
from server.rag.index.embedder.document_embedder import document_embedder


class AsyncTextParser:
    ADD_BATCH_SIZE = 30
    DELETE_BATCH_SIZE = 100

    def __init__(self) -> None:
        self.max_chunk_length = MAX_CHUNK_LENGTH
        self.chunk_overlap = CHUNK_OVERLAP
        self.doc_source = FROM_LOCAL_FILE
        self.sqlite_db_path = f"{SQLITE_DB_DIR}/{SQLITE_DB_NAME}"
        self.distributed_lock = diskcache_lock

    async def update_doc_status(self, doc_id: int, doc_status: int) -> None:
        logger.info(
            f"[FILE_CONTENT] update_doc_status, doc_id: {doc_id}, doc_status: {doc_status}"
        )
        timestamp = int(time.time())
        async with aiosqlite.connect(self.sqlite_db_path) as db:
            await db.execute("PRAGMA journal_mode=WAL;")

            try:
                with self.distributed_lock.lock():
                    await db.execute(
                        "UPDATE t_local_file_tab SET doc_status = ?, mtime = ? WHERE id = ?",
                        (doc_status, timestamp, doc_id))
                    await db.commit()
            except Exception as e:
                logger.error(f"Process distributed_lock exception: {e}")

    async def add_local_file_chunk(self, file_id: int,
                                   chunk_text_vec: List[str],
                                   start_index: int) -> None:
        logger.info(
            f"[FILE_CONTENT] add_local_file_chunk, file_id: {file_id}, start_index: {start_index}"
        )
        timestamp = int(time.time())
        async with aiosqlite.connect(self.sqlite_db_path) as db:
            await db.execute("PRAGMA journal_mode=WAL;")

            chunks_to_add = []
            chunk_index = start_index + 1
            for chunk in chunk_text_vec:
                chunks_to_add.append((file_id, chunk_index, chunk, len(chunk),
                                      timestamp, timestamp))
                chunk_index += 1

            try:
                with self.distributed_lock.lock():
                    await db.executemany(
                        "INSERT INTO t_local_file_chunk_tab (file_id, chunk_index, content, content_length, ctime, mtime) VALUES (?, ?, ?, ?, ?, ?)",
                        chunks_to_add)
                    await db.commit()
            except Exception as e:
                logger.error(f"Process distributed_lock exception: {e}")

    async def add_content(self, doc_id: int, content: str, url: str) -> None:
        if not content:
            await self.update_doc_status(doc_id, LOCAL_FILE_PROCESS_FAILED)
            return

        begin_time = int(time.time())
        logger.info(
            f"[FILE_CONTENT] add_content begin, doc_id: {doc_id}, begin_time: {begin_time}"
        )
        await self.update_doc_status(doc_id, LOCAL_FILE_PARSING)

        text_splitter_obj = MarkdownTextSplitter(
            chunk_size=self.max_chunk_length, chunk_overlap=self.chunk_overlap)
        chunk_text_vec = text_splitter_obj.split_text(content)
        await self.update_doc_status(doc_id, LOCAL_FILE_PARSING_COMPLETED)

        embedding_id_vec = []
        for start in range(0, len(chunk_text_vec), self.ADD_BATCH_SIZE):
            batch = chunk_text_vec[start:start + self.ADD_BATCH_SIZE]

            await self.add_local_file_chunk(doc_id, batch, start)

            try:
                with self.distributed_lock.lock():
                    ret = await document_embedder.aadd_local_file_embedding(
                        doc_id, url, batch, self.doc_source)
                    if ret:
                        embedding_id_vec.extend(ret)
            except Exception as e:
                logger.error(f"Process distributed_lock exception: {e}")

        if embedding_id_vec:
            await self.update_doc_status(doc_id, LOCAL_FILE_EMBEDDED)

            try:
                timestamp = int(time.time())
                with self.distributed_lock.lock():

                    async with aiosqlite.connect(self.sqlite_db_path) as db:
                        await db.execute("PRAGMA journal_mode=WAL;")

                        await db.execute(
                            "INSERT INTO t_doc_embedding_map_tab (doc_id, doc_source, embedding_id_list, ctime, mtime) VALUES (?, ?, ?, ?, ?)",
                            (doc_id, self.doc_source,
                             json.dumps(embedding_id_vec), timestamp,
                             timestamp))
                        await db.commit()
            except Exception as e:
                logger.error(f"Process distributed_lock exception: {e}")

        end_time = int(time.time())
        timecost = end_time - begin_time
        logger.warning(
            f"[FILE_CONTENT] add_content end, end_time: {end_time}, timecost: {timecost}"
        )

    async def delete_content(self, doc_id: int) -> None:
        begin_time = int(time.time())
        logger.info(
            f"[FILE_CONTENT] delete_content begin, doc_id: {doc_id}, begin_time: {begin_time}"
        )

        # Delete embeddings associated with doc ID
        await self._delete_embedding_doc(doc_id)

        async with aiosqlite.connect(self.sqlite_db_path) as db:
            await db.execute("PRAGMA journal_mode=WAL;")

            try:
                with self.distributed_lock.lock():
                    await db.execute(
                        f"DELETE FROM t_local_file_tab WHERE id = ?",
                        (doc_id, ))
                    await db.commit()
            except Exception as e:
                logger.error(f"Process distributed_lock exception: {e}")

        end_time = int(time.time())
        timecost = end_time - begin_time
        logger.warning(
            f"[FILE_CONTENT] delete_content end, end_time: {end_time}, timecost: {timecost}"
        )

    async def _delete_embedding_doc(self, doc_id: int) -> None:
        logger.info(f"[CRAWL_CONTENT] _delete_embedding_doc, doc_id: {doc_id}")
        async with aiosqlite.connect(self.sqlite_db_path) as db:
            await db.execute("PRAGMA journal_mode=WAL;")

            cursor = await db.execute(
                f"SELECT embedding_id_list FROM t_doc_embedding_map_tab WHERE doc_source = ? and doc_id = ?",
                [self.doc_source, doc_id])
            rows = await cursor.fetchall()
            # Parse embedding_id_list and flatten the list
            embedding_id_vec = [
                id for row in rows for id in json.loads(row[0])
            ]

            try:
                with self.distributed_lock.lock():
                    # Delete records from t_doc_embedding_map_tab
                    await db.execute(
                        f"DELETE FROM t_doc_embedding_map_tab WHERE doc_source = ? and doc_id = ?",
                        (self.doc_source, doc_id))
                    await db.commit()
            except Exception as e:
                logger.error(f"Process distributed_lock exception: {e}")

            try:
                for start in range(0, len(embedding_id_vec),
                                   self.DELETE_BATCH_SIZE):
                    batch = embedding_id_vec[start:start +
                                             self.DELETE_BATCH_SIZE]
                    if batch:
                        with self.distributed_lock.lock():
                            logger.info(
                                f"[CRAWL_CONTENT] _delete_embedding_doc, document_embedder.delete_document_embedding: {batch}"
                            )
                            document_embedder.delete_document_embedding(batch)
                            # await document_embedder.adelete_document_embedding(batch)
            except Exception as e:
                logger.error(f"Process distributed_lock exception: {e}")
