import os
import fitz
from llama_parse import LlamaParse
from server.logger.logger_config import my_logger as logger
from server.rag.index.parser.file_loader.pymupdf_rag import to_markdown

USE_LLAMA_PARSE = int(os.getenv('USE_LLAMA_PARSE'))
LLAMA_CLOUD_API_KEY = os.getenv('LLAMA_CLOUD_API_KEY')


class AsyncPdfLoader:
    def __init__(self, file_path: str) -> None:
        logger.info(f"[FILE LOADER] init pdf, file_path: '{file_path}'")
        self.file_path = file_path

    async def get_content(self) -> str:
        try:
            content = ''

            if USE_LLAMA_PARSE:
                parser = LlamaParse(
                    api_key=LLAMA_CLOUD_API_KEY,
                    result_type="markdown",
                )

                text_vec = []

                import nest_asyncio
                nest_asyncio.apply()

                documents = parser.load_data(self.file_path)
                for doc in documents:
                    text_vec.append(doc.text)
                content = "\n\n".join(text_vec)
            else:
                with fitz.open(self.file_path) as doc:
                    content = to_markdown(doc)

            if not content:
                logger.warning(f"file_path: '{self.file_path}' is empty!")
            return content
        except Exception as e:
            logger.error(f"get_content is failed, exception: {e}")
            return ''
