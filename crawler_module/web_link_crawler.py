# coding=utf-8
import aiohttp
import aiosqlite
import asyncio
import time
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from utils.logger_config import my_logger as logger


class AsyncCrawlerSiteLink:

    def __init__(self, base_url, sqlite_db_path, max_requests, version, distributed_lock):
        logger.info(f"[CRAWL_LINK] init, base_url:'{base_url}', version:{version}")
        self.base_url = self.normalize_url(base_url)
        self.sqlite_db_path = sqlite_db_path
        self.visited_urls = set()
        self.semaphore = asyncio.Semaphore(max_requests)
        self.domain = urlparse(self.base_url).netloc
        self.version = version
        self.distributed_lock = distributed_lock
        self.count = 0
        self.batch_urls_queue = asyncio.Queue()
        self.queue_lock = asyncio.Lock()
        self.batch_size = max_requests * 4

    async def fetch_page(self, session, url):
        logger.info(f"[CRAWL_LINK] fetch_page, url:'{url}'")
        async with self.semaphore:
            try:
                async with session.get(url) as response:
                    return await response.text()
            except Exception as e:
                logger.error(f"[CRAWL_LINK] fetch_page, Error fetching {url}: {e}")
                return None

    def is_same_domain(self, url):
        return urlparse(url).netloc == urlparse(self.base_url).netloc

    def normalize_url(self, url):
        return url.split('#')[0].rstrip('/')

    async def add_url_to_queue(self, url):
        logger.info(f"[CRAWL_LINK] add_url_to_queue, url:'{url}'")
        batch_urls = []
        async with self.queue_lock:
            await self.batch_urls_queue.put(url)
            if self.batch_urls_queue.qsize() >= self.batch_size:
                batch_urls = await self.process_batch_urls()

        if batch_urls:
            # Process batch_urls to separate existing and new URLs
            existing_urls, new_urls = await self.check_urls_existence(batch_urls)
            # Update existing URLs and insert new URLs
            await self.update_and_insert_urls(existing_urls, new_urls)

    async def process_batch_urls(self):
        logger.info(f"[CRAWL_LINK] process_batch_urls")
        # Lock is already acquired in add_url_to_queue
        batch_urls = []
        while not self.batch_urls_queue.empty():
            batch_urls.append(await self.batch_urls_queue.get())
        return batch_urls

    async def check_urls_existence(self, batch_urls):
        logger.info(f"[CRAWL_LINK] check_urls_existence, batch_urls:{batch_urls}")
        """
        Check which URLs exist in the database and separate them from new URLs.
        """
        # Prepare query to check existence
        placeholders = ', '.join(['?'] * len(batch_urls))
        query = f"SELECT url, id FROM t_raw_tab WHERE url IN ({placeholders})"

        async with aiosqlite.connect(self.sqlite_db_path) as db:
            await db.execute("PRAGMA journal_mode=WAL;")

            cursor = await db.execute(query, batch_urls)
            existing_records = await cursor.fetchall()

        # Convert list of existing records into a dict for easy lookup
        existing_urls = {record[0]: record[1] for record in existing_records}
        # Identify new URLs
        new_urls = [url for url in batch_urls if url not in existing_urls]
        return existing_urls, new_urls

    async def update_and_insert_urls(self, existing_urls, new_urls):
        logger.info(f"[CRAWL_LINK] update_and_insert_urls, existing_urls:{existing_urls}, new_urls:{new_urls}")
        """
        Update existing URLs and insert new URLs into the database in a single operation.
        """
        timestamp = int(time.time())
        update_query = "UPDATE t_raw_tab SET doc_status = ?, version = ?, mtime = ? WHERE id = ?"
        insert_query = "INSERT INTO t_raw_tab (domain, url, content, content_length, doc_status, version, ctime, mtime) VALUES (?, ?, ?, ?, ?, ?, ?, ?)"

        # Prepare data for updating existing URLs
        updates = [(1, self.version, timestamp, url_id) for url_id in existing_urls.values()]
        # Prepare data for inserting new URLs
        inserts = [(self.domain, url, '', 0, 1, self.version, timestamp, timestamp) for url in new_urls]

        async with aiosqlite.connect(self.sqlite_db_path) as db:
            await db.execute("PRAGMA journal_mode=WAL;")

            try:
                with self.distributed_lock.lock():
                    if updates:
                        await db.executemany(update_query, updates)
                    if inserts:
                        await db.executemany(insert_query, inserts)
                    await db.commit()
            except Exception as e:
                logger.error(f"process distributed_lock exception:{e}")

    async def save_link_to_db(self, url):
        logger.info(f"[CRAWL_LINK] save_link_to_db, url:'{url}'")
        await self.add_url_to_queue(url)

    async def parse_link(self, session, html, url):
        logger.info(f"[CRAWL_LINK] parse_link, url:'{url}'")
        link_vec = await self.extract_link(html, url)
        await self.save_link_to_db(url)
        for full_link in link_vec:
            normalized_link = self.normalize_url(full_link)
            if normalized_link not in self.visited_urls and self.is_same_domain(normalized_link):
                self.visited_urls.add(normalized_link)
                await self.crawl_link(session, normalized_link)

    async def extract_link(self, html, url):
        logger.info(f"[CRAWL_LINK] extrack_link, url:'{url}'")
        try:
            soup = BeautifulSoup(html, 'html.parser')
            # Extract links within the body
            link_vec = {urljoin(url, a['href']) for a in soup.find_all('a', href=True)} if soup.body else set()
            return link_vec
        except Exception as e:
            logger.error(f"Error processing content from {url}: {str(e)}")
            return set()

    async def crawl_link(self, session, url):
        self.count += 1
        logger.info(f"[CRAWL_LINK] craw_link, url:'{url}', count:{self.count}")
        html = await self.fetch_page(session, url)
        if html:
            await self.parse_link(session, html, url)

    async def update_site_domain_status(self, domain_status):
        logger.info(f"[CRAWL_LINK] update_site_domain_status, domain_status:{domain_status}")
        timestamp = int(time.time())
        async with aiosqlite.connect(self.sqlite_db_path) as db:
            await db.execute("PRAGMA journal_mode=WAL;")

            try:
                with self.distributed_lock.lock():
                    await db.execute(
                        "UPDATE t_domain_tab SET domain_status = ?, mtime = ? WHERE domain = ?",
                        (domain_status, timestamp, self.domain)
                    )
                    await db.commit()
            except Exception as e:
                logger.error(f"process distributed_lock exception:{e}")

    async def mark_expired_link(self):
        logger.info(f"[CRAWL_LINK] mark_expired_link")
        timestamp = int(time.time())
        async with aiosqlite.connect(self.sqlite_db_path) as db:
            # Enable WAL mode for better concurrency
            await db.execute("PRAGMA journal_mode=WAL;")

            try:
                with self.distributed_lock.lock():
                    # Update doc_status to 5 for URLs that are not currently marked as status 1
                    await db.execute(
                        "UPDATE t_raw_tab SET doc_status = ?, mtime = ? WHERE domain = ? AND doc_status != 1",
                        (5, timestamp, self.domain)
                    )
                    await db.commit()
            except Exception as e:
                logger.error(f"process distributed_lock exception:{e}")

    async def run(self):
        begin_time = int(time.time())
        logger.info(f"[CRAWL_LINK] run begin! base_url:{self.base_url}', begin_time:{begin_time}")
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/52.0.2743.116 Safari/537.36",
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
            logger.info(f"[CRAW_LINK process the remaining urls:{batch_urls} in batch_urls_queue")

            existing_urls, new_urls = await self.check_urls_existence(batch_urls)
            await self.update_and_insert_urls(existing_urls, new_urls)

        await self.mark_expired_link()
        await self.update_site_domain_status(2)
        end_time = int(time.time())
        timecost = end_time - begin_time
        logger.info(f"[CRAWL_LINK] run end! base_url:'{self.base_url}', end_time:{end_time}, timecost:{timecost}")

