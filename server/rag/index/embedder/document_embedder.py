import json
import os
import time
from typing import List, Tuple, Dict, Optional
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain.schema.document import Document
from server.constant.constants import (OPENAI_EMBEDDING_MODEL_NAME,
                                       ZHIPUAI_EMBEDDING_MODEL_NAME,
                                       CHROMA_DB_DIR, CHROMA_COLLECTION_NAME,
                                       OLLAMA_EMBEDDING_MODEL_NAME)
from server.logger.logger_config import my_logger as logger
from server.rag.index.embedder.zhipuai_embedder import ZhipuAIEmbeddings


class DocumentEmbedder:
    BATCH_SIZE = 30

    def __init__(self) -> None:
        self.llm_name = os.getenv('LLM_NAME')
        if self.llm_name == 'OpenAI':
            embeddings = OpenAIEmbeddings(
                openai_api_key=os.getenv('OPENAI_API_KEY'),
                model=OPENAI_EMBEDDING_MODEL_NAME)
        elif self.llm_name == 'ZhipuAI':
            embeddings = ZhipuAIEmbeddings(
                api_key=os.getenv('ZHIPUAI_API_KEY'),
                model=ZHIPUAI_EMBEDDING_MODEL_NAME)
        elif self.llm_name == 'Ollama':
            base_url = os.getenv('OLLAMA_BASE_URL')
            embeddings = OllamaEmbeddings(base_url=base_url,
                                          model=OLLAMA_EMBEDDING_MODEL_NAME)
        elif self.llm_name in ['DeepSeek', 'Moonshot']:
            # DeepSeek and Moonshot use ZhipuAI's Embedding API
            embeddings = ZhipuAIEmbeddings(
                api_key=os.getenv('ZHIPUAI_API_KEY'),
                model=ZHIPUAI_EMBEDDING_MODEL_NAME)
        else:
            raise ValueError(
                f"Unsupported LLM_NAME '{self.llm_name}'. Must be in ['OpenAI', 'ZhipuAI', 'Ollama', 'DeepSeek', 'Moonshot']."
            )

        collection_name = CHROMA_COLLECTION_NAME
        persist_directory = CHROMA_DB_DIR
        logger.info(
            f"[DOC_EMBEDDER] init, collection_name: '{collection_name}', persist_directory: '{persist_directory}', llm_name: '{self.llm_name}'"
        )
        collection_metadata = {"hnsw:space": "cosine"}
        self.chroma_vector = Chroma(collection_name=collection_name,
                                    embedding_function=embeddings,
                                    persist_directory=persist_directory,
                                    collection_metadata=collection_metadata)

    async def aadd_document_embedding(
        self, data: List[Tuple[int, str, List[str]]], doc_source: int
    ) -> Tuple[List[Tuple[int, int, str, int, int]], List[Tuple[int, int]]]:
        records_to_add: List[Tuple[int, int, str, int, int]] = []
        records_to_update: List[Tuple[int, int]] = []
        for item in data:
            documents_to_add: List[Document] = []
            timestamp = int(time.time())
            doc_id, url, chunk_text_vec = item
            for part_index, part_content in enumerate(chunk_text_vec):
                metadata: Dict[str, str] = {
                    "source": url,
                    "id": f"{doc_source}-{doc_id}-part{part_index}"
                }
                doc = Document(page_content=part_content, metadata=metadata)
                documents_to_add.append(doc)

            if documents_to_add:
                embedding_id_vec: List[str] = []
                for start in range(0, len(documents_to_add), self.BATCH_SIZE):
                    batch = documents_to_add[start:start + self.BATCH_SIZE]
                    ret = await self.chroma_vector.aadd_documents(batch)
                    embedding_id_vec.extend(ret)
                logger.info(
                    f"[DOC_EMBEDDER] doc_id={doc_id}, url={url}, doc_source={doc_source}, added {len(documents_to_add)} chunk parts to Chroma, embedding_id_vec={embedding_id_vec}"
                )
                records_to_add.append(
                    (doc_id, doc_source, json.dumps(embedding_id_vec),
                     timestamp, timestamp))
                records_to_update.append((timestamp, doc_id))

        return records_to_add, records_to_update

    async def aadd_local_file_embedding(self, doc_id: int, url: str,
                                        chunk_text_vec: List[str],
                                        doc_source: int) -> List[str]:
        file_documents_to_add = []
        for part_index, part_content in enumerate(chunk_text_vec):
            metadata: Dict[str, str] = {
                "source": url,
                "id": f"{doc_source}-{doc_id}-part{part_index}"
            }
            doc = Document(page_content=part_content, metadata=metadata)
            file_documents_to_add.append(doc)

        if file_documents_to_add:
            embedding_id_vec = await self.chroma_vector.aadd_documents(
                file_documents_to_add)
            logger.info(
                f"[DOC_EMBEDDER] doc_id={doc_id}, url={url}, doc_source={doc_source}, added {len(file_documents_to_add)} chunk parts to Chroma, embedding_id_vec={embedding_id_vec}"
            )
            return embedding_id_vec
        else:
            return []

    async def adelete_document_embedding(
            self, embedding_id_vec: List[str]) -> Optional[bool]:
        for start in range(0, len(embedding_id_vec), self.BATCH_SIZE):
            batch = embedding_id_vec[start:start + self.BATCH_SIZE]
            await self.chroma_vector.adelete(batch)
        logger.info(
            f"[DOC_EMBEDDER] Deleted {len(embedding_id_vec)} embeddings from Chroma."
        )

    def delete_document_embedding(self, embedding_id_vec: List[str]) -> None:
        for start in range(0, len(embedding_id_vec), self.BATCH_SIZE):
            batch = embedding_id_vec[start:start + self.BATCH_SIZE]
            self.chroma_vector.delete(batch)
        logger.info(
            f"[DOC_EMBEDDER] Deleted {len(embedding_id_vec)} embeddings from Chroma."
        )


document_embedder = DocumentEmbedder()
