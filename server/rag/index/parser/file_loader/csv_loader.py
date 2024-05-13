import pandas as pd
from server.logger.logger_config import my_logger as logger


class AsyncCsvLoader:
    def __init__(self, file_path: str) -> None:
        logger.info(f"[FILE LOADER] init csv, file_path: '{file_path}'")
        self.file_path = file_path

    async def get_content(self) -> str:
        try:
            content = ''

            # Load the CSV file into a DataFrame
            df = pd.read_csv(self.file_path)
            # Convert the DataFrame to a Markdown string
            content = df.to_markdown(index=False)

            if not content:
                logger.warnning(f"file_path: '{self.file_path}' is empty!")
            return content
        except Exception as e:
            logger.error(f"get_content is failed, exception: {e}")
            return ''
