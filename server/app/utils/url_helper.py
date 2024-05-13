from urllib.parse import urlparse


def is_valid_url(url: str) -> bool:
    """Check if the provided string is a valid URL."""
    parsed_url = urlparse(url)
    return bool(parsed_url.scheme) and bool(parsed_url.netloc)


def is_same_domain(url1: str, url2: str) -> bool:
    return urlparse(url1).netloc == urlparse(url2).netloc


def normalize_url(url: str) -> str:
    return url.split('#')[0].rstrip('/')
