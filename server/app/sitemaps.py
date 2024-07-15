import asyncio
from functools import wraps
import json
from threading import Thread
import time
from typing import Callable, Dict, Any, List
from urllib.parse import urlparse
from flask import Blueprint, request
from server.app.utils.decorators import token_required
from server.app.utils.sqlite_client import get_db_connection
from server.app.utils.diskcache_lock import diskcache_lock
from server.app.utils.url_helper import is_valid_url
from server.constant.constants import (ADD_SITEMAP_CONTENT,
                                       DELETE_SITEMAP_CONTENT,
                                       UPDATE_SITEMAP_CONTENT,
                                       DOMAIN_PROCESSING, FROM_SITEMAP_URL)
from server.logger.logger_config import my_logger as logger
from server.rag.index.parser.html_parser.web_link_crawler import AsyncCrawlerSiteLink
from server.rag.index.parser.html_parser.web_content_crawler import AsyncCrawlerSiteContent

sitemaps_bp = Blueprint('sitemaps',
                        __name__,
                        url_prefix='/open_kf_api/sitemaps')


def async_crawl_link_task(site: str, version: int) -> None:
    """Start the crawl link task in an asyncio event loop."""
    logger.info(f"Create crawler_link")
    crawler_link = AsyncCrawlerSiteLink(
        base_url=site,
        version=version,
    )
    logger.info(
        f"async_crawl_link_task begin!, site: '{site}', version: {version}")
    asyncio.run(crawler_link.run())
    logger.info(
        f"async_crawl_link_task end!, site: '{site}', version: {version}")


@sitemaps_bp.route('/submit_crawl_site', methods=['POST'])
@token_required
def submit_crawl_site():
    """Submit a site for crawling."""
    data = request.json
    site = data.get('site')
    timestamp = data.get('timestamp')

    if not site or not timestamp:
        return {
            'retcode': -20000,
            'message': 'site and timestamp are required',
            'data': {}
        }

    if not is_valid_url(site):
        logger.error(f"site: '{site} is not a valid URL!")
        return {
            'retcode': -20001,
            'message': f"site: '{site}' is not a valid URL",
            'data': {}
        }

    domain = urlparse(site).netloc
    logger.info(f"domain is '{domain}'")
    conn = None
    try:
        timestamp = int(timestamp)
        conn = get_db_connection()
        cur = conn.cursor()

        # Check if the domain exists in the database
        cur.execute(
            "SELECT id, version FROM t_sitemap_domain_tab WHERE domain = ?",
            (domain, ))
        domain_info = cur.fetchone()

        if domain_info and timestamp <= domain_info["version"]:
            return {
                'retcode': -20001,
                'message':
                f'New timestamp: {timestamp} must be greater than the current version: {domain_info["version"]}.',
                'data': {}
            }

        try:
            with diskcache_lock.lock():
                if domain_info:
                    domain_id, version = domain_info
                    # Update domain record
                    cur.execute(
                        "UPDATE t_sitemap_domain_tab SET version = ?, domain_status = 1, mtime=? WHERE id = ?",
                        (timestamp, int(time.time()), domain_id))
                else:
                    # Insert new domain record
                    cur.execute(
                        "INSERT INTO t_sitemap_domain_tab (domain, domain_status, version, ctime, mtime) VALUES (?, 1, ?, ?, ?)",
                        (domain, timestamp, int(time.time()), int(
                            time.time())))

                conn.commit()
        except Exception as e:
            logger.error(f"Process discache_lock exception:{e}")
            return {
                'retcode': -30000,
                'message': f'An error occurred: {e}',
                'data': {}
            }

        # Start the asynchronous crawl task
        Thread(target=async_crawl_link_task, args=(site, timestamp)).start()

        return {
            'retcode': 0,
            'message': 'Site submitted successfully for crawling.',
            'data': {}
        }
    except Exception as e:
        return {
            'retcode': -30000,
            'message': f'An error occurred: {e}',
            'data': {}
        }
    finally:
        if conn:
            conn.close()


@sitemaps_bp.route('/get_crawl_site_info', methods=['POST'])
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
                logger.error(f"site: '{site}' is not a valid URL!")
                return {
                    'retcode': -20001,
                    'message': f"site: '{site}' is not a valid URL",
                    'data': {}
                }
            domain = urlparse(site).netloc
            logger.info(f"Searching for domain: '{domain}'")
            cur.execute("SELECT * FROM t_sitemap_domain_tab WHERE domain = ?",
                        (domain, ))
        else:
            logger.info("Fetching information for all sites.")
            cur.execute("SELECT * FROM t_sitemap_domain_tab")

        rows = cur.fetchall()
        if rows:
            sites_info = [dict(row) for row in rows]
            return {
                'retcode': 0,
                'message': 'Success',
                'data': {
                    'sites_info': sites_info
                }
            }
        else:
            return {
                'retcode': -20001,
                'message': 'No site information found',
                'data': {}
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


@sitemaps_bp.route('/get_crawl_url_list', methods=['POST'])
@token_required
def get_crawl_url_list():
    """Fetch the list of URLs and their status information. If the site is specified and valid, returns information for that site. Returns an error if the site is specified but invalid."""
    data = request.json
    site = data.get('site', None)  # Make site an optional parameter

    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        response_data = {'url_list': []}

        if site is not None:
            if not is_valid_url(site):
                logger.error(f"Provided site: '{site}' is not a valid URL.")
                return {
                    'retcode': -20001,
                    'message': f"Provided site: '{site}' is not a valid URL.",
                    'data': {}
                }

            domain = urlparse(site).netloc
            logger.info(f"Fetching URL list for domain: '{domain}'")
            cur.execute(
                "SELECT domain_status FROM t_sitemap_domain_tab WHERE domain = ?",
                (domain, ))
            domain_status_row = cur.fetchone()
            if domain_status_row:
                response_data['domain_status'] = domain_status_row[
                    'domain_status']
            cur.execute(
                "SELECT id, url, content_length, doc_status, version, ctime, mtime FROM t_sitemap_url_tab WHERE domain = ?",
                (domain, ))
        else:
            logger.info("Fetching URL list for all domains.")
            cur.execute(
                "SELECT id, url, content_length, doc_status, version, ctime, mtime FROM t_sitemap_url_tab"
            )

        rows = cur.fetchall()
        response_data['url_list'] = [dict(row) for row in rows]
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


def async_crawl_content_task(domain_list: List[str], url_dict: Dict[int, str],
                             task_type: int) -> None:
    """
    Starts the asynchronous crawl and embedding process for a list of urls.

    task_type:
      1 - add_content
      2 - delete_content
      3 - update_content
    """
    """Start the crawl content task in an asyncio event loop."""
    logger.info(
        f"async_crawl_content_task begin! domain_lsit: {domain_list}, url_dict: {url_dict}, task_type: {task_type}"
    )
    crawler_content = AsyncCrawlerSiteContent(domain_list=domain_list,
                                              doc_source=FROM_SITEMAP_URL)

    # Run the crawler
    if task_type == ADD_SITEMAP_CONTENT:
        asyncio.run(crawler_content.add_content(url_dict))
    elif task_type == DELETE_SITEMAP_CONTENT:
        asyncio.run(crawler_content.delete_content(url_dict))
    elif task_type == UPDATE_SITEMAP_CONTENT:
        asyncio.run(crawler_content.update_content(url_dict))
    logger.info(
        f"async_crawl_content_task end!, domain_list: {domain_list}', task_type: {task_type}"
    )


# Define the type for a generic Flask view function
FlaskViewFunction = Callable[..., Dict[str, Any]]


def check_crawl_content_task(f: FlaskViewFunction) -> FlaskViewFunction:
    @wraps(f)
    def decorated_function(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        data = request.json
        id_list = data.get('id_list')

        if not id_list or not isinstance(id_list, list) or len(id_list) == 0:
            return {
                'retcode': -20000,
                'message': 'Invalid or missing id_list parameter'
            }

        conn = None
        try:
            conn = get_db_connection()
            cur = conn.cursor()

            placeholders = ', '.join(['?'] * len(id_list))
            cur.execute(
                f"SELECT id, domain, url FROM t_sitemap_url_tab WHERE id IN ({placeholders})",
                id_list)
            rows = cur.fetchall()

            if len(rows) != len(id_list):
                missing_ids = set(id_list) - set(row[0] for row in rows)
                return {
                    'retcode': -20001,
                    'message':
                    f'The following ids do not exist: {missing_ids}',
                    'data': {}
                }

            url_dict = {row["id"]: row["url"] for row in rows}
            domain_list = list(set(row["domain"] for row in rows))
            logger.info(f"domain_list is {domain_list}")

            # Store domain and url_dict in request for further use
            request.domain_list = domain_list
            request.url_dict = url_dict

            # Check and update domain_status in t_sitemap_domain_tab for all domains in domain_list
            timestamp = int(time.time())
            for domain in domain_list:
                cur.execute(
                    "SELECT domain_status FROM t_sitemap_domain_tab WHERE domain = ?",
                    (domain, ))
                domain_info = cur.fetchone()
                # if domain_info and domain_info["domain_status"] < 3:
                if domain_info and domain_info[
                        "domain_status"] < DOMAIN_PROCESSING:
                    try:
                        with diskcache_lock.lock():
                            cur.execute(
                                "UPDATE t_sitemap_domain_tab SET domain_status = ?, mtime = ? WHERE domain = ?",
                                (DOMAIN_PROCESSING, timestamp, domain))
                            conn.commit()
                            logger.info(
                                f"Updated domain_status to {DOMAIN_PROCESSING} for domain: '{domain}'"
                            )
                    except Exception as e:
                        logger.error(f"Process discache_lock exception:{e}")
                        return {
                            'retcode': -30000,
                            'message': f'An error occurred: {e}',
                            'data': {}
                        }
        except Exception as e:
            return {
                'retcode': -30000,
                'message': f'An error occurred: {e}',
                'data': {}
            }
        finally:
            if conn:
                conn.close()
        return f(*args, **kwargs)

    return decorated_function


@sitemaps_bp.route('/add_crawl_url_list', methods=['POST'])
@check_crawl_content_task
@token_required
def add_crawl_url_list():
    domain_list = request.domain_list
    url_dict = request.url_dict
    # Use threading to avoid blocking the Flask application
    Thread(target=async_crawl_content_task,
           args=(domain_list, url_dict, ADD_SITEMAP_CONTENT)).start()
    return {
        'retcode': 0,
        'message': 'Started processing the URL list.',
        'data': {}
    }


@sitemaps_bp.route('/delete_crawl_url_list', methods=['POST'])
@check_crawl_content_task
@token_required
def delete_crawl_url_list():
    domain_list = request.domain_list
    url_dict = request.url_dict
    # Use threading to avoid blocking the Flask application
    Thread(target=async_crawl_content_task,
           args=(domain_list, url_dict, DELETE_SITEMAP_CONTENT)).start()
    return {
        'retcode': 0,
        'message': 'Started deleting the URL list embeddings.',
        'data': {}
    }


@sitemaps_bp.route('/update_crawl_url_list', methods=['POST'])
@check_crawl_content_task
@token_required
def update_crawl_url_list():
    domain_list = request.domain_list
    url_dict = request.url_dict
    # Use threading to avoid blocking the Flask application
    Thread(target=async_crawl_content_task,
           args=(domain_list, url_dict, UPDATE_SITEMAP_CONTENT)).start()
    return {
        'retcode': 0,
        'message': 'Started updating the URL list embeddings.',
        'data': {}
    }


@sitemaps_bp.route('/get_crawl_url_sub_content_list', methods=['POST'])
@token_required
def get_crawl_url_sub_content_list():
    data = request.json
    url_id = data.get('id')
    page = data.get('page')
    page_size = data.get('page_size')

    # Validate mandatory parameters
    if None in (url_id, page, page_size):
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

        # Retrieve the content from the database
        cur.execute('SELECT content FROM t_sitemap_url_tab WHERE id = ?',
                    (url_id, ))
        row = cur.fetchone()
        if not row:
            return {
                'retcode': -30000,
                'message': 'Content not found',
                'data': {}
            }

        content = row['content']
        content_vec = json.loads(content)

        # Calculate pagination details
        total_count = len(content_vec)
        start_index = (page - 1) * page_size
        end_index = start_index + page_size
        if start_index > 0 and start_index >= total_count:
            return {
                'retcode': -20002,
                'message': 'Page number out of range',
                'data': {}
            }

        # Slice the content vector to get the sub-content list for the current page
        sub_content_list = [{
            "index": start_index + index + 1,
            "content": part,
            "content_length": len(part)
        } for index, part in enumerate(content_vec[start_index:end_index],
                                       start=start_index)]

        return {
            "retcode": 0,
            "message": "success",
            "data": {
                "total_count": total_count,
                "sub_content_list": sub_content_list
            }
        }
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return {'retcode': -30001, 'message': 'Database exception', 'data': {}}
    finally:
        if conn:
            conn.close()
