from server.logger.logger_config import my_logger as logger


class AsyncMdLoader:
    def __init__(self, file_path: str) -> None:
        logger.info(f"[FILE LOADER] init md, file_path: '{file_path}'")
        self.file_path = file_path

    async def get_content(self) -> str:
        try:
            content = ''
            with open(self.file_path, 'r', encoding='utf-8') as fd:
                content = fd.read()

            if not content:
                logger.warning(f"file_path: '{self.file_path}' is empty!")
            return content
        except Exception as e:
            logger.error(f"get_content is failed, exception: {e}")
            return ''
