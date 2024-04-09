# coding=utf-8
import aiohttp
import aiosqlite
import asyncio
import json
import os
import time
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from utils.logger_config import my_logger as logger


class AsyncCrawlerSiteContent:

    def __init__(self, domain_list, sqlite_db_path, max_requests, max_embedding_input, document_embedder_obj, redis_lock):
        logger.info(f"[CRAWL_CONTENT] init, domain_list:{domain_list}")
        self.domain_list = domain_list
        self.sqlite_db_path = sqlite_db_path
        self.semaphore = asyncio.Semaphore(max_requests)
        self.max_embedding_input = max_embedding_input
        self.document_embedder_obj = document_embedder_obj
        self.redis_lock = redis_lock
        self.count = 0
        self.batch_size = max_requests * 2

    async def fetch_page(self, session, doc_id, url):
        logger.info(f"[CRAWL_CONTENT] fetch_page, doc_id:{doc_id}, url:'{url}'")
        async with self.semaphore:
            try:
                async with session.get(url) as response:
                    return await response.text()
            except Exception as e:
                logger.error(f"[CRAWL_CONTENT] fetch_page, Error fetching doc_id:{doc_id}, url:'{url}', exception:{e}")
                return None

    async def parse_content(self, html, url):
        logger.info(f"[CRAWL_CONTENT] parse_content, url:'{url}'")
        chunk_text_vec = []
        max_token_len = self.max_embedding_input
        try:
            soup = BeautifulSoup(html, 'html.parser')

            # Extract title text
            title_text = soup.title.string.strip() if soup.title else ""
            curr_chunk = title_text + '\n'
            curr_len = len(title_text)

            # Remove <script> and <style> elements before extracting text
            for script_or_style in soup(['script', 'style', 'head', 'meta', 'link']):
                script_or_style.decompose()

            last_element_text = ''
            if soup.body:
                for element in soup.body.descendants:
                    element_text = ''
                    if element.name == 'a' and element.get('href'):
                        href = element['href']
                        # Check if href is an external link
                        if href.startswith('http://') or href.startswith('https://'):
                            text = element.get_text(strip=True)
                            # Embed the link directly in the text
                            element_text = f"{text}[{href}]"
                        else:
                            # For relative URLs, just include the text
                            element_text = element.get_text(strip=True)
                    elif element.string and not element.name:
                        # Append non-link text
                        element_text = element.string.strip()

                    if element_text:
                        element_len = len(element_text)
                        if element_len > max_token_len:
                            logger.warning(f"[CRAWL_CONTENG] parse_conteng, url:'{url}', warning element_len={element_len}")

                        if curr_len + element_len <= max_token_len:
                            curr_chunk += element_text
                            curr_chunk += '\n'
                            curr_len += element_len + 1
                        else:
                            chunk_text_vec.append(curr_chunk)
                            curr_chunk = last_element_text + '\n' + element_text + '\n'
                            curr_len = len(curr_chunk)
                        last_element_text = element_text

                if curr_chunk:
                    chunk_text_vec.append(curr_chunk)

            return chunk_text_vec
        except Exception as e:
            logger.error(f"[CRAWL_CONTENG] parse_content, url:'{url}', Error processing content exception:{str(e)}")
            return []

    async def crawl_content(self, session, doc_id, url, fetched_contents):
        self.count += 1
        logger.info(f"[CRAWL_CONTENT] crawl_content, doc_id:{doc_id}, url:'{url}', count:{self.count}")
        html = await self.fetch_page(session, doc_id, url)
        if html:
            chunk_text_vec = await self.parse_content(html, url)
            fetched_contents[doc_id] = chunk_text_vec

    async def update_doc_status(self, doc_id_list, doc_status):
        logger.info(f"[CRAWL_CONTENT] update_doc_status, doc_id_list:{doc_id_list}, doc_status:{doc_status}")
        timestamp = int(time.time())
        async with aiosqlite.connect(self.sqlite_db_path) as db:
            # Enable WAL mode for better concurrency
            await db.execute("PRAGMA journal_mode=WAL;")

            if await self.redis_lock.aacquire_lock():
                try:
                    await db.execute(
                        "UPDATE t_raw_tab SET doc_status = ?, mtime = ? WHERE id IN ({placeholders})".format(
                            placeholders=','.join(['?' for _ in doc_id_list])
                        ),
                        [doc_status, timestamp] + doc_id_list
                    )
                    await db.commit()
                finally:
                    await self.redis_lock.arelease_lock()

    async def fetch_existing_contents(self, doc_id_list):
        """
        Fetch existing contents from the database for the provided doc_id_list.
        """
        logger.info(f"[CRAWL_CONTENT] fetch_existing_contents, doc_id_list:{doc_id_list}")
        query = "SELECT id, content FROM t_raw_tab WHERE id IN ({})".format(', '.join('?' for _ in doc_id_list))
        async with aiosqlite.connect(self.sqlite_db_path) as db:
            # Enable WAL mode for better concurrency
            await db.execute("PRAGMA journal_mode=WAL;")

            cursor = await db.execute(query, doc_id_list)
            results = await cursor.fetchall()
            return dict(results)

    def compare_contents(self, existing_contents, fetched_contents):
        """
        Compare existing contents from the database with newly fetched contents.

        :param existing_contents: A dictionary of document IDs to their content as stored in the database.
        :param fetched_contents: A dictionary of document IDs to their newly fetched content.
        :return: A tuple containing a dictionary of updated contents and a set of unchanged document IDs.
        """
        logger.info(f"[CRAWL_CONTENT] compare_contents")
        updated_contents = {}
        unchanged_doc_ids = []
        for doc_id, chunk_text_vec in fetched_contents.items():
            new_content = json.dumps(chunk_text_vec)
            old_content = existing_contents.get(doc_id)
            if new_content != old_content:
                updated_contents[doc_id] = chunk_text_vec
            else:
                unchanged_doc_ids.append(doc_id)
        return updated_contents, unchanged_doc_ids

    async def compare_and_update_contents(self, url_dict, fetched_contents):
        """
        Compare fetched contents with existing ones and perform updates if necessary.
        """
        logger.info(f"[CRAWL_CONTENT] compare_and_update_contents, url_dict:{url_dict}")
        existing_contents = await self.fetch_existing_contents(list(fetched_contents.keys()))
        updated_contents, unchanged_doc_ids = self.compare_contents(existing_contents, fetched_contents)

        # Process updated contents: delete old embeddings, insert new ones, and update DB records
        if updated_contents:
            await self.process_updated_contents(updated_contents, url_dict)

        # For unchanged contents, simply update their status in the database
        if unchanged_doc_ids:
            await self.update_unchanged_contents_status(unchanged_doc_ids)

    async def process_updated_contents(self, updated_contents, url_dict):
        """
        Handle the processing of updated contents including updating the content details in the database,
        deleting old embeddings, inserting new ones, and finally updating database records in batch.
        """
        logger.info(f"[CRAWL_CONTENT] process_updated_contents, updating {len(updated_contents)} items.")
        # Prepare update queries for updating content details in t_raw_tab
        content_update_queries = []
        timestamp = int(time.time())
        for doc_id, chunk_text_vec in updated_contents.items():
            content_json = json.dumps(chunk_text_vec)
            content_length = len(content_json)
            content_update_queries.append((content_json, content_length, 3, timestamp, doc_id))

        # Lock to ensure database operations are atomic
        if await self.redis_lock.aacquire_lock():
            try:
                async with aiosqlite.connect(self.sqlite_db_path) as db:
                    # Enable WAL mode for better concurrency
                    await db.execute("PRAGMA journal_mode=WAL;")

                    # Update content details in t_raw_tab
                    await db.executemany(
                        "UPDATE t_raw_tab SET content = ?, content_length = ?, doc_status = ?, mtime = ? WHERE id = ?",
                        content_update_queries
                    )
                    await db.commit()
            finally:
                await self.redis_lock.arelease_lock()

        # Delete old embeddings
        doc_id_list = list(updated_contents.keys())
        await self.delete_embedding_doc(doc_id_list)

        # Prepare data for updating embeddings and database records
        data_for_embedding = [(doc_id, url_dict[doc_id], chunk_text_vec) for doc_id, chunk_text_vec in updated_contents.items()]
        if await self.redis_lock.aacquire_lock():
            try:
                records_to_add, records_to_update = await self.document_embedder_obj.aadd_content_embedding(data_for_embedding)
                # Insert new embedding records and update t_raw_tab doc_status to 4
                async with aiosqlite.connect(self.sqlite_db_path) as db:
                    # Enable WAL mode for better concurrency
                    await db.execute("PRAGMA journal_mode=WAL;")

                    if records_to_add:
                        await db.executemany(
                            "INSERT INTO t_doc_embedding_map_tab (doc_id, embedding_id_list, ctime, mtime) VALUES (?, ?, ?, ?)",
                            records_to_add
                        )
                    if records_to_update:
                        await db.executemany("UPDATE t_raw_tab SET doc_status = ?, mtime = ? WHERE id = ?", records_to_update)
                    await db.commit()
            finally:
                await self.redis_lock.arelease_lock()

    async def update_unchanged_contents_status(self, unchanged_doc_ids):
        """
        Update the status of unchanged contents in the database to reflect they have been processed.
        """
        logger.info(f"[CRAWL_CONTENT] update_unchanged_contents_status, unchanged_doc_ids:{unchanged_doc_ids}")
        async with aiosqlite.connect(self.sqlite_db_path) as db:
            # Enable WAL mode for better concurrency
            await db.execute("PRAGMA journal_mode=WAL;")

            if await self.redis_lock.aacquire_lock():
                try:
                    async with aiosqlite.connect(self.sqlite_db_path) as db:
                        await db.execute("UPDATE t_raw_tab SET doc_status = 4 WHERE id IN ({})".format(', '.join('?' for _ in unchanged_doc_ids)), unchanged_doc_ids)
                        await db.commit()
                finally:
                    await self.redis_lock.arelease_lock()

    async def add_content(self, url_dict):
        """Begin processing URLs from url_dict in batches for add."""
        begin_time = int(time.time())
        logger.info(f"[CRAWL_CONTENT] add_content begin!, begin_time:{begin_time}, url_dict:{url_dict}")

        # Divide url_dict into batches, each with batch_size URLs
        batches = [dict(list(url_dict.items())[i:i + self.batch_size]) for i in range(0, len(url_dict), self.batch_size)]

        # Process each batch asynchronously
        for batch in batches:
            await self.process_add_batch(batch)

        await self.check_and_update_domain_status()
        end_time = int(time.time())
        timecost = end_time - begin_time
        logger.info(f"[CRAWL_CONTENT] add_content end!, end_time:{end_time}, timecost:{timecost}")

    async def process_add_batch(self, batch):
        """Process a single batch of URLs for add."""
        logger.info(f"[CRAWL_CONTENT] process_add_batch, batch:{batch}")
        fetched_contents = {}
        doc_id_list = list(batch.keys())

        # Update document status before fetching the page
        await self.update_doc_status(doc_id_list, 2)

        # Asynchronously fetch page content for all URLs in the batch
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/52.0.2743.116 Safari/537.36",
            "Accept": "*/*",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive"
        }
        async with aiohttp.ClientSession(headers=headers) as session:
            task_vec = [self.crawl_content(session, doc_id, batch[doc_id], fetched_contents) for doc_id in batch]
            await asyncio.gather(*task_vec)

        # Compare and update content after fetching
        await self.compare_and_update_contents(batch, fetched_contents)

    async def delete_embedding_doc(self, doc_id_vec):
        logger.info(f"[CRAWL_CONTENT] delete_embedding_doc, doc_id_vec:{doc_id_vec}")
        doc_id_tuple = tuple(doc_id_vec)
        placeholder = ','.join('?' * len(doc_id_vec))  # Create placeholders
        async with aiosqlite.connect(self.sqlite_db_path) as db:
            # Enable WAL mode for better concurrency
            await db.execute("PRAGMA journal_mode=WAL;")

            cursor = await db.execute(f"SELECT embedding_id_list FROM t_doc_embedding_map_tab WHERE doc_id IN ({placeholder})", doc_id_tuple)
            rows = await cursor.fetchall()
            # Parse embedding_id_list and flatten the list
            embedding_id_vec = [id for row in rows for id in json.loads(row[0])]

            if await self.redis_lock.aacquire_lock():
                try:
                    if embedding_id_vec:
                        logger.info(f"[CRAWL_CONTENT] delete_embedding_doc, document_embedder_obj.delete_content_embedding:{embedding_id_vec}")
                        self.document_embedder_obj.delete_content_embedding(embedding_id_vec)

                    # Delete records from t_doc_embedding_map_tab
                    await db.execute(f"DELETE FROM t_doc_embedding_map_tab WHERE doc_id IN ({placeholder})", doc_id_tuple)
                    await db.commit()
                finally:
                    await self.redis_lock.arelease_lock()

    async def delete_content(self, url_dict, delete_raw_table=True):
        """Begin processing URLs from url_dict in batches for deletion."""
        begin_time = int(time.time())
        logger.info(f"[CRAWL_CONTENT] delete_content begin, url_dict:{url_dict}, delete_raw_table:{delete_raw_table}, begin_time:{begin_time}")

        # Divide url_dict into batches, each with a specified number of URLs
        batches = [dict(list(url_dict.items())[i:i + self.batch_size]) for i in range(0, len(url_dict), self.batch_size)]

        # Process each batch asynchronously
        for batch in batches:
            await self.process_delete_batch(batch, delete_raw_table)

        await self.check_and_update_domain_status()
        end_time = int(time.time())
        timecost = end_time - begin_time
        logger.info(f"[CRAWL_CONTENT] delete_content begin, delete_raw_table:{delete_raw_table}, end_time:{end_time}, timecost:{timecost}")

    async def process_delete_batch(self, batch, delete_raw_table):
        """Process a single batch of URLs for deletion."""
        logger.info(f"[CRAWL_CONTENT] process_delete_batch, batch:{batch}, delete_raw_table:{delete_raw_table}")
        doc_id_vec = list(batch.keys())
        doc_id_tuple = tuple(doc_id_vec)
        placeholder = ','.join('?' * len(doc_id_vec))  # Create placeholders

        # Delete embeddings associated with doc IDs
        await self.delete_embedding_doc(doc_id_vec)

        if delete_raw_table:
            # Delete records from t_raw_tab after deleting embeddings
            async with aiosqlite.connect(self.sqlite_db_path) as db:
                await db.execute("PRAGMA journal_mode=WAL;")
                if await self.redis_lock.aacquire_lock():
                    try:
                        await db.execute(f"DELETE FROM t_raw_tab WHERE id IN ({placeholder})", doc_id_tuple)
                        await db.commit()
                    finally:
                        await self.redis_lock.arelease_lock()

    async def update_content(self, url_dict):
        logger.info(f"[CRAWL_CONTENT] update_content begin, url_dict:{url_dict}")
        # Just copy `add_content`
        await self.add_content(url_dict)
        logger.info(f"[CRAWL_CONTENT] update_content end, url_dict:{url_dict}")

    async def check_and_update_domain_status(self):
        logger.info(f"[CRAWL_CONTENT] check_and_update_domain_status")
        async with aiosqlite.connect(self.sqlite_db_path) as db:
            # Enable WAL mode for better concurrency
            await db.execute("PRAGMA journal_mode=WAL;")

            timestamp = int(time.time())
            for domain in self.domain_list:
                # Step 1: Check current domain_status for the domain
                cursor = await db.execute("SELECT domain_status FROM t_domain_tab WHERE domain = ?", (domain,))
                row = await cursor.fetchone()
                if row and row[0] != 4:
                    # Step 2: Check if all URLs for the domain have doc_status >= 4
                    cursor = await db.execute(
                        "SELECT COUNT(*) FROM t_raw_tab WHERE domain = ? AND doc_status < 4", (domain,))
                    count_row = await cursor.fetchone()
                    if count_row[0] == 0:  # If no records have doc_status < 4
                        if await self.redis_lock.aacquire_lock():
                            try:
                                # Step 3: Update domain_status to 4 in t_domain_tab
                                await db.execute(
                                    "UPDATE t_domain_tab SET domain_status = ?, mtime = ? WHERE domain = ?", (4, timestamp, domain))
                                await db.commit()
                            finally:
                                await self.redis_lock.arelease_lock()
                        logger.info(f"[CRAWL_CONTENT] check_and_update_domain_status, Domain status updated to 4 for domain:'{domain}'")

