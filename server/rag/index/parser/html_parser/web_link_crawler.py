import aiohttp
import aiosqlite
import asyncio
import time
from typing import List, Tuple, Dict, Set, Optional
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from server.app.utils.url_helper import is_same_domain, normalize_url
from server.app.utils.diskcache_lock import diskcache_lock
from server.constant.constants import (SQLITE_DB_DIR, SQLITE_DB_NAME,
                                       MAX_CRAWL_PARALLEL_REQUEST,
                                       SITEMAP_URL_RECORDED,
                                       SITEMAP_URL_EXPIRED,
                                       DOMAIN_STATISTICS_GATHERING_COLLECTED)
from server.logger.logger_config import my_logger as logger


class AsyncCrawlerSiteLink:
    def __init__(self, base_url: str, version: int) -> None:
        logger.info(
            f"[CRAWL_LINK] init, base_url: '{base_url}', version: {version}")
        self.base_url = normalize_url(base_url)
        self.sqlite_db_path = f"{SQLITE_DB_DIR}/{SQLITE_DB_NAME}"
        self.visited_urls = set()
        self.semaphore = asyncio.Semaphore(MAX_CRAWL_PARALLEL_REQUEST)
        self.domain = urlparse(self.base_url).netloc
        self.version = version
        self.distributed_lock = diskcache_lock
        self.count = 0
        self.batch_urls_queue = asyncio.Queue()
        self.queue_lock = asyncio.Lock()
        self.batch_size = MAX_CRAWL_PARALLEL_REQUEST * 4

    async def fetch_page(self, session: aiohttp.ClientSession,
                         url: str) -> Optional[str]:
        logger.info(f"[CRAWL_LINK] fetch_page, url: '{url}'")
        async with self.semaphore:
            try:
                async with session.get(url) as response:
                    return await response.text()
            except Exception as e:
                logger.error(
                    f"[CRAWL_LINK] fetch_page, Error fetching {url}: {e}")
                return None

    async def add_url_to_queue(self, url: str) -> None:
        logger.info(f"[CRAWL_LINK] add_url_to_queue, url: '{url}'")
        batch_urls = []
        async with self.queue_lock:
            await self.batch_urls_queue.put(url)
            if self.batch_urls_queue.qsize() >= self.batch_size:
                batch_urls = await self.process_batch_urls()

        if batch_urls:
            # Process batch_urls to separate existing and new URLs
            existing_urls, new_urls = await self.check_urls_existence(
                batch_urls)
            # Update existing URLs and insert new URLs
            await self.update_and_insert_urls(existing_urls, new_urls)

    async def process_batch_urls(self) -> List[str]:
        logger.info(f"[CRAWL_LINK] process_batch_urls")
        # Lock is already acquired in add_url_to_queue
        batch_urls = []
        while not self.batch_urls_queue.empty():
            batch_urls.append(await self.batch_urls_queue.get())
        return batch_urls

    async def check_urls_existence(
            self, batch_urls: List[str]) -> Tuple[Dict[str, int], List[str]]:
        """
        Check which URLs exist in the database and separate them from new URLs.
        """
        logger.info(
            f"[CRAWL_LINK] check_urls_existence, batch_urls: {batch_urls}")
        # Prepare sql to check existence
        placeholders = ', '.join(['?'] * len(batch_urls))
        sql = f"SELECT url, id FROM t_sitemap_url_tab WHERE doc_status = 4 and url IN ({placeholders})"

        async with aiosqlite.connect(self.sqlite_db_path) as db:
            await db.execute("PRAGMA journal_mode=WAL;")

            cursor = await db.execute(sql, batch_urls)
            existing_records = await cursor.fetchall()

        # Convert list of existing records into a dict for easy lookup
        existing_urls = {record[0]: record[1] for record in existing_records}
        # Identify new URLs
        new_urls = [url for url in batch_urls if url not in existing_urls]
        return existing_urls, new_urls

    async def update_and_insert_urls(self, existing_urls: Dict[str, int],
                                     new_urls: List[str]) -> None:
        """
        Update existing URLs and insert new URLs into the database in a single operation.
        """
        logger.info(
            f"[CRAWL_LINK] update_and_insert_urls, existing_urls: {existing_urls}, new_urls: {new_urls}"
        )
        timestamp = int(time.time())
        update_sql = "UPDATE t_sitemap_url_tab SET doc_status = ?, version = ?, mtime = ? WHERE id = ?"
        insert_sql = "INSERT INTO t_sitemap_url_tab (domain, url, content, content_length, content_md5, doc_status, version, ctime, mtime) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"

        # Prepare data for updating existing URLs
        updates: List[Tuple[int, int, int, int]] = [
            (SITEMAP_URL_RECORDED, self.version, timestamp, url_id)
            for url_id in existing_urls.values()
        ]
        # Prepare data for inserting new URLs
        inserts: List[Tuple[str, str, str, int, str, int, int, int, int]] = [
            (self.domain, url, '[]', 0, '', SITEMAP_URL_RECORDED, self.version,
             timestamp, timestamp) for url in new_urls
        ]

        async with aiosqlite.connect(self.sqlite_db_path) as db:
            await db.execute("PRAGMA journal_mode=WAL;")

            try:
                with self.distributed_lock.lock():
                    if updates:
                        await db.executemany(update_sql, updates)
                    if inserts:
                        await db.executemany(insert_sql, inserts)
                    await db.commit()
            except Exception as e:
                logger.error(f"process distributed_lock exception:{e}")

    async def save_link_to_db(self, url: str) -> None:
        logger.info(f"[CRAWL_LINK] save_link_to_db, url: '{url}'")
        await self.add_url_to_queue(url)

    async def parse_link(self, session: aiohttp.ClientSession, html_text: str,
                         url: str) -> None:
        logger.info(f"[CRAWL_LINK] parse_link, url: '{url}'")
        link_set = await self.extract_link(html_text, url)
        await self.save_link_to_db(url)
        for full_link in link_set:
            normalized_link = normalize_url(full_link)
            if normalized_link not in self.visited_urls and is_same_domain(
                    self.base_url, normalized_link):
                self.visited_urls.add(normalized_link)
                await self.crawl_link(session, normalized_link)

    async def extract_link(self, html_text: str, url: str) -> Set[str]:
        logger.info(f"[CRAWL_LINK] extrack_link, url: '{url}'")
        try:
            soup = BeautifulSoup(html_text, 'html.parser')
            # Extract links within the body
            link_set = {
                urljoin(url, a['href'])
                for a in soup.find_all('a', href=True)
            } if soup.body else set()
            return link_set
        except Exception as e:
            logger.error(f"Error processing content from {url}: {str(e)}")
            return set()

    async def crawl_link(self, session: aiohttp.ClientSession,
                         url: str) -> None:
        self.count += 1
        logger.info(
            f"[CRAWL_LINK] craw_link, url: '{url}', count: {self.count}")
        html_text = await self.fetch_page(session, url)
        if html_text:
            await self.parse_link(session, html_text, url)

    async def update_site_domain_status(self, domain_status: int) -> None:
        logger.info(
            f"[CRAWL_LINK] update_site_domain_status, domain_status: {domain_status}"
        )
        timestamp = int(time.time())
        async with aiosqlite.connect(self.sqlite_db_path) as db:
            await db.execute("PRAGMA journal_mode=WAL;")

            try:
                with self.distributed_lock.lock():
                    await db.execute(
                        "UPDATE t_sitemap_domain_tab SET domain_status = ?, mtime = ? WHERE domain = ?",
                        (domain_status, timestamp, self.domain))
                    await db.commit()
            except Exception as e:
                logger.error(f"process distributed_lock exception:{e}")

    async def mark_expired_link(self) -> None:
        logger.info(f"[CRAWL_LINK] mark_expired_link")
        timestamp = int(time.time())
        async with aiosqlite.connect(self.sqlite_db_path) as db:
            # Enable WAL mode for better concurrency
            await db.execute("PRAGMA journal_mode=WAL;")

            try:
                with self.distributed_lock.lock():
                    # Update doc_status to `SITEMAP_URL_EXPIRED` for URLs that are not currently marked as status `SITEMAP_URL_RECORDED`
                    await db.execute(
                        "UPDATE t_sitemap_url_tab SET doc_status = ?, mtime = ? WHERE domain = ? AND doc_status != ?",
                        (SITEMAP_URL_EXPIRED, timestamp, self.domain,
                         SITEMAP_URL_RECORDED))
                    await db.commit()
            except Exception as e:
                logger.error(f"process distributed_lock exception:{e}")

    async def run(self) -> None:
        begin_time = int(time.time())
        logger.info(
            f"[CRAWL_LINK] run begin! base_url: '{self.base_url}', begin_time: {begin_time}"
        )
        headers = {
            "User-Agent":
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/52.0.2743.116 Safari/537.36",
            "Accept": "*/*",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive"
        }
        async with aiohttp.ClientSession(headers=headers) as session:
            self.visited_urls.add(self.base_url)
            await self.crawl_link(session, self.base_url)

        if self.batch_urls_queue.qsize() > 0:
            # Process the remaining urls in batch_urls_queue
            batch_urls = []
            while not self.batch_urls_queue.empty():
                batch_urls.append(await self.batch_urls_queue.get())
            logger.info(
                f"[CRAW_LINK process the remaining urls:{batch_urls} in batch_urls_queue"
            )

            existing_urls, new_urls = await self.check_urls_existence(
                batch_urls)
            await self.update_and_insert_urls(existing_urls, new_urls)

        await self.mark_expired_link()
        await self.update_site_domain_status(
            DOMAIN_STATISTICS_GATHERING_COLLECTED)
        end_time = int(time.time())
        timecost = end_time - begin_time
        logger.warning(
            f"[CRAWL_LINK] run end! base_url: '{self.base_url}', end_time: {end_time}, timecost: {timecost}"
        )
