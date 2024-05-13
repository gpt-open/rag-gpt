import html2text
import mammoth
from server.logger.logger_config import my_logger as logger


class AsyncDocxLoader:
    def __init__(self, file_path: str) -> None:
        logger.info(f"[FILE LOADER] init docx, file_path: '{file_path}'")
        self.file_path = file_path

    async def get_content(self) -> str:
        try:
            content = ''
            html_text = ''
            with open(self.file_path, 'rb') as fd:
                result = mammoth.convert_to_html(fd)
                html_text = result.value
                messages = result.messages
                if messages:
                    logger.warning(f"Read file_path: '{self.file_path}', messages: {messages}")

                if html_text:
                    # Create an html2text converter
                    h = html2text.HTML2Text()
                    h.ignore_images = True
                    content = h.handle(html_text)
                else:
                    logger.warnning(f"file_path: '{self.file_path}', convert_to_html is empty!")

            if not content:
                logger.warnning(f"file_path: '{self.file_path}' is empty!")
            return content
        except Exception as e:
            logger.error(f"get_content is failed, exception: {e}")
            return ''
