import pandas as pd
from tabulate import tabulate
from server.logger.logger_config import my_logger as logger


class AsyncXlsxLoader:
    def __init__(self, file_path: str) -> None:
        logger.info(f"[FILE LOADER] init xlsx, file_path: '{file_path}'")
        self.file_path = file_path

    async def get_content(self) -> str:
        try:
            content = ''

            # Load all sheets from the Excel file
            sheets_dict = pd.read_excel(self.file_path, sheet_name=None)
            # This will hold the markdown content for all sheets
            markdown_content = []

            # Load all sheets from the Excel file
            sheets_dict = pd.read_excel(self.file_path, sheet_name=None)
            # This will hold the markdown content for all sheets
            markdown_content = []

            # Process each sheet in the workbook
            for sheet_name, df in sheets_dict.items():
                # Add a header for each sheet in the Markdown output
                markdown_content.append(f"# {sheet_name}\n")

                # Convert the DataFrame to a Markdown string using DataFrame.to_markdown()
                markdown_str = df.to_markdown(index=False)
                markdown_content.append(markdown_str)
                markdown_content.append("\n")  # Add a newline for spacing between sheets
   
            if markdown_content:
                content = "\n".join(markdown_content)

            if not content:
                logger.warnning(f"file_path: '{self.file_path}' is empty!")
            return content
        except Exception as e:
            logger.error(f"get_content is failed, exception: {e}")
            return ''
