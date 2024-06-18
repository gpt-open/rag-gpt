import os
from llama_parse import LlamaParse
import pandas as pd
from server.logger.logger_config import my_logger as logger

USE_LLAMA_PARSE = int(os.getenv('USE_LLAMA_PARSE'))
LLAMA_CLOUD_API_KEY = os.getenv('LLAMA_CLOUD_API_KEY')


class AsyncCsvLoader:
    def __init__(self, file_path: str) -> None:
        logger.info(f"[FILE LOADER] init csv, file_path: '{file_path}'")
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
                # Load the CSV file into a DataFrame
                df = pd.read_csv(self.file_path)
                # Convert the DataFrame to a Markdown string
                content = df.to_markdown(index=False)

                if not content:
                    logger.warning(f"file_path: '{self.file_path}' is empty!")
            return content
        except Exception as e:
            logger.error(f"get_content is failed, exception: {e}")
            return ''
