from typing import List
from langchain_core.embeddings.embeddings import Embeddings
from zhipuai import ZhipuAI
from server.logger.logger_config import my_logger as logger


class ZhipuAIEmbeddings(Embeddings):
    def __init__(self, api_key: str, model: str = "embedding-2") -> None:
        self.client = ZhipuAI(api_key=api_key)
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Synchronously embed a list of documents."""
        embeddings = []
        for text in texts:
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=text,
                )
                embeddings.append(
                    response.data[0].embedding if response.data else [])
            except Exception as e:
                # Log the error and use an empty list as a fallback
                logger.error(
                    f"Error embedding document: {text} with error: {str(e)}")
                embeddings.append([])
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Synchronously embed a single query."""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text,
            )
            return response.data[0].embedding if response.data else []
        except Exception as e:
            # Log the error and return an empty list as a fallback
            logger.error(f"Error embedding query: {text} with error: {str(e)}")
            return []
