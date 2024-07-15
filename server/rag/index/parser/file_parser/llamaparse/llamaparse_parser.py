import json
import os
import requests
import tempfile
import time
from pathlib import Path
from typing import Any, List, Optional
from llama_parse import LlamaParse
from server.rag.index.parser.file_parser.llamaparse.file_handler import FileHandler
from server.logger.logger_config import my_logger as logger

all_elements_output_file = "all_elements.json"
chunks_output_file = "chunks.json"


class DocParser:
    def __init__(self,
                 file_handler: FileHandler,
                 language: str = "en",
                 is_download_image: bool = True) -> None:
        self.file_handler = file_handler
        self.is_download_image = is_download_image
        USE_GPT4O = int(os.getenv('USE_GPT4O'))
        if USE_GPT4O:
            self.llamaparse = LlamaParse(
                api_key=os.getenv('LLAMA_CLOUD_API_KEY'),
                gpt4o_mode=True,
                gpt4o_api_key=os.getenv('OPENAI_API_KEY'),
                result_type="json",
                language=language,
                verbose=True)
        else:
            self.llamaparse = LlamaParse(
                api_key=os.getenv('LLAMA_CLOUD_API_KEY'),
                result_type="json",
                language=language,
                verbose=True)
        logger.info(
            f"Init DocParser of llamaparse, language: '{language}', is_download_image: {is_download_image}, USE_GPT4O: {USE_GPT4O}"
        )

    def parse_file(
            self,
            filepath: Path,
            destination_folder: Path,
            include_chunking: bool = True) -> tuple[list[Any], list[Any]]:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = Path(temp_dir) / filepath.name
            self.file_handler.download_file(filepath.as_posix(),
                                            temp_file.as_posix())

            elements_file = f"{temp_dir}/{all_elements_output_file}"

            elements, chunks = self.partition_doc_to_folder(
                temp_file,
                Path(temp_dir),
                include_chunking=include_chunking,
                all_elements_output_file=elements_file)

            self.file_handler.sync_foler(temp_dir,
                                         destination_folder.as_posix())

            return elements, chunks

    def partition_doc(
        self,
        input_file: Path,
        output_dir: Path,
        include_chunking: bool = True,
    ) -> tuple[list[Any], list[Any]]:
        elements = []
        chunks = []
        try:
            import nest_asyncio
            nest_asyncio.apply()

            json_objs = self.llamaparse.get_json_result(str(input_file))
            job_id = json_objs[0]["job_id"]
            elements = json_objs[0]["pages"]
            job_metadata = json_objs[0]["job_metadata"]
            logger.info(
                f"For inpput_file: '{input_file}', job_id is'{job_id}', job_metatdata is {job_metadata}"
            )

            if self.is_download_image:
                """
                TODO:
                To enhance the efficiency of image downloading, the following optimizations could be considered:
                1. Handle image downloads through asynchronous tasks to improve response times.
                2. Implement concurrent downloads to make effective use of resources and accelerate the download process.
                """
                for page_item in elements:
                    images = page_item["images"]
                    for image_item in images:
                        image_name = image_item["name"]
                        logger.info(
                            f"For inpput_file: '{input_file}', downloading image: '{image_name}'"
                        )
                        download_image(job_id, image_name,
                                       output_dir.as_posix())

            if include_chunking:
                """
                TODO:
                The current chunking strategy treats each page as a separate chunk. Future optimizations might include:
                1. Evaluating whether adjacent pages can be merged into a single chunk.
                2. Considering whether it's necessary to split a single page into multiple chunks.
                """
                filename = input_file.name
                file_extension = input_file.suffix
                for page_item in elements:
                    page_number = page_item["page"]
                    chunk_item = {
                        "chunk_text": page_item["md"],
                        "metadata": {
                            "filename": filename,
                            "filetype": f"application/{file_extension[1:]}",
                            "last_modified_timestamp": int(time.time()),
                            "beginning_page": page_number,
                            "ending_page": page_number
                        }
                    }
                    chunks.append(chunk_item)
        except Exception as e:
            logger.error(
                f"Parsing file: '{input_file}' is failed, exception: {e}")

        return elements, chunks

    def partition_doc_to_folder(
        self,
        input_file: Path,
        output_dir: Path,
        all_elements_output_file: str,
        include_chunking: bool = True,
    ) -> tuple[list[Any], list[Any]]:
        elements, chunks = self.partition_doc(input_file, output_dir,
                                              include_chunking)

        elements_output_file = output_dir / all_elements_output_file
        elements_to_json(elements, elements_output_file.as_posix())
        elements_to_json(chunks, (output_dir / chunks_output_file).as_posix())

        return elements, chunks


def elements_to_json(
    elements: List[Any],
    filename: Optional[str] = None,
    indent: int = 4,
    encoding: str = "utf-8",
) -> Optional[str]:
    """
    Saves a list of elements to a JSON file if filename is specified.
    Otherwise, return the list of elements as a string.
    """
    # -- serialize `elements` as a JSON array (str) --
    json_str = json.dumps(elements, indent=indent, sort_keys=False)
    if filename is not None:
        with open(filename, "w", encoding=encoding) as f:
            f.write(json_str)
        return None
    return json_str


def download_image(job_id: str, image_name: str, output_dir: str) -> None:
    url = f"https://api.cloud.llamaindex.ai/api/parsing/job/{job_id}/result/image/{image_name}"
    headers = {
        'Authorization': f'Bearer {os.getenv("LLAMA_CLOUD_API_KEY")}',
        'Accept': 'application/json',
        'Content-Type': 'multipart/form-data'
    }
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            with open(f'{output_dir}/{image_name}', 'wb') as f:
                f.write(response.content)
        else:
            logger.error(
                f"Failed to retrieve '{image_name}', status_code: {response.status_code}, text: {response.text}"
            )
    except Exception as e:
        logger.error(f"Download '{image_name}' failed, error: {e}")
