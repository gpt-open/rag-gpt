from bs4 import BeautifulSoup
import html2text
from server.logger.logger_config import my_logger as logger


class AsyncHtmlLoader:
    def __init__(self, file_path: str) -> None:
        logger.info(f"[FILE LOADER] init html, file_path: '{file_path}'")
        self.file_path = file_path

    async def get_content(self) -> str:
        try:
            content = ''
            with open(self.file_path, 'r') as fd:
                html_text = fd.read()

                # Use BeautifulSoup to parse HTML content
                soup = BeautifulSoup(html_text, 'html.parser')
                body_content = soup.find('body')

                # Create an html2text converter
                h = html2text.HTML2Text()
                content = h.handle(str(body_content))

            if not content:
                logger.warning(f"file_path: '{self.file_path}' is empty!")
            return content
        except Exception as e:
            logger.error(f"get_content is failed, exception: {e}")
            return ''
