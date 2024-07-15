import aiohttp
import aiosqlite
import asyncio
import json
import os
import re
import time
from typing import List, Tuple, Dict, Optional
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import html2text
from server.app.utils.hash import generate_md5
from server.constant.constants import (SQLITE_DB_DIR, SQLITE_DB_NAME,
                                       MAX_CRAWL_PARALLEL_REQUEST,
                                       MAX_CHUNK_LENGTH, CHUNK_OVERLAP,
                                       FROM_SITEMAP_URL)
from server.logger.logger_config import my_logger as logger
from server.rag.index.chunk.markdown_splitter import MarkdownTextSplitter
from server.rag.index.embedder.document_embedder import document_embedder
from server.app.utils.diskcache_lock import diskcache_lock


def add_base_url_to_links(text: str, base_url: str) -> str:
    # Define the regex pattern for Markdown links
    link_pattern = re.compile(r'\[([^\]]+)\]\((?!http)([^)]+)\)')

    # Function to replace links
    def replace_link(match):
        text = match.group(1)
        link = match.group(2)
        # Check if the link already contains a complete protocol or special protocol like mailto:
        if not link.startswith(
            ('http://', 'https://', '#', 'mailto:', 'tel:')):
            # Only process relative links that start with '/'
            if link.startswith('/'):
                # Append the base URL in front of the link
                link = base_url + link
        return f'[{text}]({link})'

    # Replace all relative links in the text
    processed_content = link_pattern.sub(replace_link, text)
    return processed_content


class AsyncCrawlerSiteContent:
    def __init__(self, domain_list: List[str], doc_source: int) -> None:
        logger.info(
            f"[CRAWL_CONTENT] init, domain_list: {domain_list}, doc_source: {doc_source}"
        )
        self.domain_list = domain_list
        self.doc_source = doc_source
        self.sqlite_db_path = f"{SQLITE_DB_DIR}/{SQLITE_DB_NAME}"
        self.semaphore = asyncio.Semaphore(MAX_CRAWL_PARALLEL_REQUEST)
        self.max_chunk_length = MAX_CHUNK_LENGTH
        self.chunk_overlap = CHUNK_OVERLAP
        self.distributed_lock = diskcache_lock
        self.count = 0
        self.batch_size = MAX_CRAWL_PARALLEL_REQUEST * 2

    async def fetch_page(self, session: aiohttp.ClientSession, doc_id: int,
                         url: str) -> Optional[str]:
        logger.info(
            f"[CRAWL_CONTENT] fetch_page, doc_id: {doc_id}, url: '{url}'")
        async with self.semaphore:
            try:
                async with session.get(url) as response:
                    return await response.text()
            except Exception as e:
                logger.error(
                    f"[CRAWL_CONTENT] fetch_page, Error fetching doc_id: {doc_id}, url: '{url}', exception: {e}"
                )
                await self.update_doc_status([doc_id], 0)
                return None

    async def parse_content(self, html_text: str, url: str) -> List[str]:
        logger.info(f"[CRAWL_CONTENT] parse_content, url: '{url}'")
        try:
            # Use BeautifulSoup to parse HTML content
            soup = BeautifulSoup(html_text, 'html.parser')

            # Remove all the tags that are not meaningful for the extraction.
            SCAPE_TAGS = ["nav", "footer", "aside", "script", "style"]
            [tag.decompose() for tag in soup.find_all(SCAPE_TAGS)]

            body_content = soup.find('body')

            # Create an html2text converter
            h = html2text.HTML2Text()
            markdown_content = h.handle(str(body_content))

            # Retrieve the base URL
            parsed_url = urlparse(url)
            base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
            processed_markdown = add_base_url_to_links(markdown_content,
                                                       base_url)

            text_splitter_obj = MarkdownTextSplitter(
                chunk_size=self.max_chunk_length,
                chunk_overlap=self.chunk_overlap)
            chunk_text_vec = text_splitter_obj.split_text(processed_markdown)
            return chunk_text_vec
        except Exception as e:
            logger.error(
                f"[CRAWL_CONTENG] parse_content, url:'{url}', Error processing content exception: {e}"
            )
            return []

    async def crawl_content(self, session: aiohttp.ClientSession, doc_id: int,
                            url: str,
                            fetched_contents: Dict[int, List[str]]) -> None:
        self.count += 1
        logger.info(
            f"[CRAWL_CONTENT] crawl_content, doc_id: {doc_id}, url: '{url}', count: {self.count}"
        )
        html_text = await self.fetch_page(session, doc_id, url)
        if html_text:
            chunk_text_vec = await self.parse_content(html_text, url)
            fetched_contents[doc_id] = chunk_text_vec

    async def update_doc_status(self, doc_id_list: List[int],
                                doc_status: int) -> None:
        logger.info(
            f"[CRAWL_CONTENT] update_doc_status, doc_id_list: {doc_id_list}, doc_status: {doc_status}"
        )
        timestamp = int(time.time())
        async with aiosqlite.connect(self.sqlite_db_path) as db:
            await db.execute("PRAGMA journal_mode=WAL;")

            try:
                with self.distributed_lock.lock():
                    if self.doc_source == FROM_SITEMAP_URL:
                        await db.execute(
                            "UPDATE t_sitemap_url_tab SET doc_status = ?, mtime = ? WHERE id IN ({placeholders})"
                            .format(placeholders=','.join(
                                ['?' for _ in doc_id_list])),
                            [doc_status, timestamp] + doc_id_list)
                    else:
                        await db.execute(
                            "UPDATE t_isolated_url_tab SET doc_status = ?, mtime = ? WHERE id IN ({placeholders})"
                            .format(placeholders=','.join(
                                ['?' for _ in doc_id_list])),
                            [doc_status, timestamp] + doc_id_list)
                    await db.commit()
            except Exception as e:
                logger.error(f"process distributed_lock exception: {e}")

    async def get_existing_content_md5(
            self, doc_id_list: List[int]) -> Dict[int, str]:
        """
        Fetch existing content_md5 from the database for the provided doc_id_list.
        """
        logger.info(
            f"[CRAWL_CONTENT] get_existing_content_md5, doc_id_list: {doc_id_list}"
        )
        sql = ''
        if self.doc_source == FROM_SITEMAP_URL:
            sql = "SELECT id, content_md5 FROM t_sitemap_url_tab WHERE id IN ({})".format(
                ', '.join('?' for _ in doc_id_list))
        else:
            sql = "SELECT id, content_md5 FROM t_isolated_url_tab WHERE id IN ({})".format(
                ', '.join('?' for _ in doc_id_list))

        async with aiosqlite.connect(self.sqlite_db_path) as db:
            await db.execute("PRAGMA journal_mode=WAL;")

            cursor = await db.execute(sql, doc_id_list)
            results = await cursor.fetchall()
            return dict(results)

    def compare_contents(
        self, existing_contents_md5: Dict[int, str],
        fetched_contents: Dict[int, List[str]]
    ) -> Tuple[Dict[int, List[str]], List[int]]:
        """
        Compare existing contents from the database with newly fetched contents.

        :param existing_contents_md5: A dictionary of document IDs to their content_md5 as stored in the database.
        :param fetched_contents: A dictionary of document IDs to their newly fetched content.
        :return: A tuple containing a dictionary of updated contents and a set of unchanged document IDs.
        """
        logger.info(f"[CRAWL_CONTENT] compare_contents")
        updated_contents = {}
        unchanged_doc_ids = []
        for doc_id, chunk_text_vec in fetched_contents.items():
            new_content = json.dumps(chunk_text_vec)
            new_content_md5 = generate_md5(new_content.encode('utf-8'))
            old_content_md5 = existing_contents_md5.get(doc_id)
            if new_content_md5 != old_content_md5:
                updated_contents[doc_id] = chunk_text_vec
            else:
                unchanged_doc_ids.append(doc_id)
        return updated_contents, unchanged_doc_ids

    async def compare_and_update_contents(
            self, url_dict: Dict[int, str],
            fetched_contents: Dict[int, List[str]]) -> None:
        """
        Compare fetched contents with existing ones and perform updates if necessary.
        """
        logger.info(
            f"[CRAWL_CONTENT] compare_and_update_contents, url_dict: {url_dict}"
        )
        existing_contents_md5 = await self.get_existing_content_md5(
            list(fetched_contents.keys()))
        updated_contents, unchanged_doc_ids = self.compare_contents(
            existing_contents_md5, fetched_contents)

        # Process updated contents: delete old embeddings, insert new ones, and update DB records
        if updated_contents:
            await self.process_updated_contents(updated_contents, url_dict)

        # For unchanged contents, simply update their status in the database
        if unchanged_doc_ids:
            await self.update_unchanged_contents_status(unchanged_doc_ids)

    async def process_updated_contents(self, updated_contents: Dict[int,
                                                                    List[str]],
                                       url_dict: Dict[int, str]) -> None:
        """
        Handle the processing of updated contents including updating the content details in the database,
        deleting old embeddings, inserting new ones, and finally updating database records in batch.
        """
        logger.info(
            f"[CRAWL_CONTENT] process_updated_contents, updating {len(updated_contents)} items."
        )
        content_update_queries: List[Tuple[str, int, str, int, int, int]] = []
        timestamp = int(time.time())
        for doc_id, chunk_text_vec in updated_contents.items():
            content = json.dumps(chunk_text_vec)
            content_length = len(content)
            content_md5 = generate_md5(content.encode('utf-8'))
            content_update_queries.append(
                (content, content_length, content_md5, 3, timestamp, doc_id))

        # Lock to ensure database operations are atomic
        async with aiosqlite.connect(self.sqlite_db_path) as db:
            await db.execute("PRAGMA journal_mode=WAL;")

            try:
                with self.distributed_lock.lock():
                    if self.doc_source == FROM_SITEMAP_URL:
                        await db.executemany(
                            "UPDATE t_sitemap_url_tab SET content = ?, content_length = ?, content_md5 = ?, doc_status = ?, mtime = ? WHERE id = ?",
                            content_update_queries)
                    else:
                        await db.executemany(
                            "UPDATE t_isolated_url_tab SET content = ?, content_length = ?, content_md5 = ?, doc_status = ?, mtime = ? WHERE id = ?",
                            content_update_queries)
                    await db.commit()
            except Exception as e:
                logger.error(f"process distributed_lock exception: {e}")

        # Delete old embeddings
        doc_id_list = list(updated_contents.keys())
        await self._delete_embedding_doc(doc_id_list)

        # Prepare data for updating embeddings and database records
        data_for_embedding = [
            (doc_id, url_dict[doc_id], chunk_text_vec)
            for doc_id, chunk_text_vec in updated_contents.items()
        ]
        try:
            with self.distributed_lock.lock():
                records_to_add, records_to_update = await document_embedder.aadd_document_embedding(
                    data_for_embedding, self.doc_source)
                async with aiosqlite.connect(self.sqlite_db_path) as db:
                    await db.execute("PRAGMA journal_mode=WAL;")

                    if records_to_add:
                        await db.executemany(
                            "INSERT INTO t_doc_embedding_map_tab (doc_id, doc_source, embedding_id_list, ctime, mtime) VALUES (?, ?, ?, ?, ?)",
                            records_to_add)
                    if records_to_update:
                        if self.doc_source == FROM_SITEMAP_URL:
                            await db.executemany(
                                "UPDATE t_sitemap_url_tab SET doc_status = 4, mtime = ? WHERE id = ?",
                                records_to_update)
                        else:
                            await db.executemany(
                                "UPDATE t_isolated_url_tab SET doc_status = 4, mtime = ? WHERE id = ?",
                                records_to_update)
                    await db.commit()
        except Exception as e:
            logger.error(f"process distributed_lock exception: {e}")

    async def update_unchanged_contents_status(
            self, unchanged_doc_ids: List[int]) -> None:
        """
        Update the status of unchanged contents in the database to reflect they have been processed.
        """
        logger.info(
            f"[CRAWL_CONTENT] update_unchanged_contents_status, unchanged_doc_ids: {unchanged_doc_ids}"
        )
        async with aiosqlite.connect(self.sqlite_db_path) as db:
            await db.execute("PRAGMA journal_mode=WAL;")

            try:
                with self.distributed_lock.lock():
                    async with aiosqlite.connect(self.sqlite_db_path) as db:
                        if self.doc_source == FROM_SITEMAP_URL:
                            await db.execute(
                                "UPDATE t_sitemap_url_tab SET doc_status = 4 WHERE id IN ({})"
                                .format(', '.join('?'
                                                  for _ in unchanged_doc_ids)),
                                unchanged_doc_ids)
                        else:
                            await db.execute(
                                "UPDATE t_isolated_url_tab SET doc_status = 4 WHERE id IN ({})"
                                .format(', '.join('?'
                                                  for _ in unchanged_doc_ids)),
                                unchanged_doc_ids)
                        await db.commit()
            except Exception as e:
                logger.error(f"process distributed_lock exception: {e}")

    async def add_content(self, url_dict: Dict[int, str]) -> None:
        """Begin processing URLs from url_dict in batches for add."""
        begin_time = int(time.time())
        logger.info(
            f"[CRAWL_CONTENT] add_content begin!, begin_time: {begin_time}, url_dict: {url_dict}"
        )

        # Divide url_dict into batches, each with batch_size URLs
        batches = [
            dict(list(url_dict.items())[i:i + self.batch_size])
            for i in range(0, len(url_dict), self.batch_size)
        ]

        # Process each batch asynchronously
        for batch in batches:
            await self.process_add_batch(batch)

        if self.doc_source == FROM_SITEMAP_URL:
            await self.check_and_update_domain_status()

        end_time = int(time.time())
        timecost = end_time - begin_time
        logger.warning(
            f"[CRAWL_CONTENT] add_content end!, end_time: {end_time}, timecost: {timecost}"
        )

    async def process_add_batch(self, batch: Dict[int, str]) -> None:
        """Process a single batch of URLs for add."""
        logger.info(f"[CRAWL_CONTENT] process_add_batch, batch: {batch}")
        fetched_contents = {}
        doc_id_list = list(batch.keys())

        # Update document status before fetching the page
        await self.update_doc_status(doc_id_list, 2)

        # Asynchronously fetch page content for all URLs in the batch
        headers = {
            "User-Agent":
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/52.0.2743.116 Safari/537.36",
            "Accept": "*/*",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive"
        }
        async with aiohttp.ClientSession(headers=headers) as session:
            task_vec = [
                self.crawl_content(session, doc_id, batch[doc_id],
                                   fetched_contents) for doc_id in batch
            ]
            await asyncio.gather(*task_vec)

        # Compare and update content after fetching
        await self.compare_and_update_contents(batch, fetched_contents)

    async def _delete_embedding_doc(self, doc_id_vec: List[int]) -> None:
        logger.info(
            f"[CRAWL_CONTENT] _delete_embedding_doc, doc_id_vec: {doc_id_vec}")
        placeholder = ','.join('?' * len(doc_id_vec))
        async with aiosqlite.connect(self.sqlite_db_path) as db:
            await db.execute("PRAGMA journal_mode=WAL;")

            cursor = await db.execute(
                f"SELECT embedding_id_list FROM t_doc_embedding_map_tab WHERE doc_source = ? and  doc_id IN ({placeholder})",
                [self.doc_source] + doc_id_vec)
            rows = await cursor.fetchall()
            # Parse embedding_id_list and flatten the list
            embedding_id_vec = [
                id for row in rows for id in json.loads(row[0])
            ]

            try:
                with self.distributed_lock.lock():
                    if embedding_id_vec:
                        logger.info(
                            f"[CRAWL_CONTENT] _delete_embedding_doc, document_embedder.delete_document_embedding: {embedding_id_vec}"
                        )
                        document_embedder.delete_document_embedding(
                            embedding_id_vec)
                        # await document_embedder.adelete_document_embedding(embedding_id_vec)

                    # Delete records from t_doc_embedding_map_tab
                    await db.execute(
                        f"DELETE FROM t_doc_embedding_map_tab WHERE doc_source = ? and doc_id IN ({placeholder})",
                        [self.doc_source] + doc_id_vec)
                    await db.commit()
            except Exception as e:
                logger.error(f"process distributed_lock exception: {e}")

    async def delete_content(self,
                             url_dict: Dict[int, str],
                             delete_raw_table: bool = True) -> None:
        """Begin processing URLs from url_dict in batches for deletion."""
        begin_time = int(time.time())
        logger.info(
            f"[CRAWL_CONTENT] delete_content begin, url_dict: {url_dict}, delete_raw_table: {delete_raw_table}, begin_time: {begin_time}"
        )

        # Divide url_dict into batches, each with a specified number of URLs
        batches = [
            dict(list(url_dict.items())[i:i + self.batch_size])
            for i in range(0, len(url_dict), self.batch_size)
        ]

        # Process each batch asynchronously
        for batch in batches:
            await self.process_delete_batch(batch, delete_raw_table)

        if self.doc_source == FROM_SITEMAP_URL:
            await self.check_and_update_domain_status()

        end_time = int(time.time())
        timecost = end_time - begin_time
        logger.warning(
            f"[CRAWL_CONTENT] delete_content end, delete_raw_table: {delete_raw_table}, end_time: {end_time}, timecost: {timecost}"
        )

    async def process_delete_batch(self, batch: Dict[int, str],
                                   delete_raw_table: bool) -> None:
        """Process a single batch of URLs for deletion."""
        logger.info(
            f"[CRAWL_CONTENT] process_delete_batch, batch: {batch}, delete_raw_table: {delete_raw_table}"
        )
        doc_id_vec = list(batch.keys())
        placeholder = ','.join('?' * len(doc_id_vec))

        # Delete embeddings associated with doc IDs
        await self._delete_embedding_doc(doc_id_vec)

        if delete_raw_table:
            async with aiosqlite.connect(self.sqlite_db_path) as db:
                await db.execute("PRAGMA journal_mode=WAL;")

                try:
                    with self.distributed_lock.lock():
                        if self.doc_source == FROM_SITEMAP_URL:
                            await db.execute(
                                f"DELETE FROM t_sitemap_url_tab WHERE id IN ({placeholder})",
                                doc_id_vec)
                        else:
                            await db.execute(
                                f"DELETE FROM t_isolated_url_tab WHERE id IN ({placeholder})",
                                doc_id_vec)
                        await db.commit()
                except Exception as e:
                    logger.error(f"process distributed_lock exception: {e}")

    async def update_content(self, url_dict: Dict[int, str]) -> None:
        logger.info(
            f"[CRAWL_CONTENT] update_content begin, url_dict: {url_dict}")
        # Just copy `add_content`
        await self.add_content(url_dict)
        logger.info(
            f"[CRAWL_CONTENT] update_content end, url_dict: {url_dict}")

    async def check_and_update_domain_status(self) -> None:
        logger.info(f"[CRAWL_CONTENT] check_and_update_domain_status")
        async with aiosqlite.connect(self.sqlite_db_path) as db:
            await db.execute("PRAGMA journal_mode=WAL;")

            timestamp = int(time.time())
            for domain in self.domain_list:
                # Step 1: Check current domain_status for the domain
                cursor = await db.execute(
                    "SELECT domain_status FROM t_sitemap_domain_tab WHERE domain = ?",
                    (domain, ))
                row = await cursor.fetchone()
                if row and row[0] != 4:
                    # Step 2: Check if all URLs for the domain have doc_status >= 4
                    cursor = await db.execute(
                        "SELECT COUNT(*) FROM t_sitemap_url_tab WHERE domain = ? AND doc_status < 4",
                        (domain, ))
                    count_row = await cursor.fetchone()
                    if count_row[0] == 0:  # If no records have doc_status < 4
                        try:
                            with self.distributed_lock.lock():
                                # Step 3: Update domain_status to 4 in t_sitemap_domain_tab
                                await db.execute(
                                    "UPDATE t_sitemap_domain_tab SET domain_status = ?, mtime = ? WHERE domain = ?",
                                    (4, timestamp, domain))
                                await db.commit()
                        except Exception as e:
                            logger.error(
                                f"process distributed_lock exception: {e}")
                        logger.info(
                            f"[CRAWL_CONTENT] check_and_update_domain_status, Domain status updated to 4 for domain:'{domain}'"
                        )
