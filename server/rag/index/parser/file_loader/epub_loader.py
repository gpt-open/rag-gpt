import fitz
from server.logger.logger_config import my_logger as logger
from server.rag.index.parser.file_loader.pymupdf_rag import to_markdown


class AsyncEpubLoader:
    def __init__(self, file_path: str) -> None:
        logger.info(f"[FILE LOADER] init epub, file_path: '{file_path}'")
        self.file_path = file_path

    async def get_content(self) -> str:
        try:
            content = ''
            with fitz.open(self.file_path) as doc:
                content = to_markdown(doc)

            if not content:
                logger.warning(f"file_path: '{self.file_path}' is empty!")
            return content
        except Exception as e:
            logger.error(f"get_content is failed, exception: {e}")
            return ''
