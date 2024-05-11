from typing import List, Tuple
from langchain.schema.document import Document
from server.rag.index.embedder.document_embedder import document_embedder


class VectorSearch:
    def __init__(self) -> None:
        self.vector_db = document_embedder.chroma_vector

    def max_marginal_relevance_search(self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5
    ) -> List[Tuple[Document, float]]:
        """
        Return docs selected using the maximal marginal relevance.
        Maximal marginal relevance optimizes for similarity to query AND diversity among selected documents.
        """
        ret = self.vector_db.max_marginal_relevance_search(query=query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult)
        return [(doc, 0.0) for doc in ret]

    def similarity_search_with_score(self, query: str, k: int = 4) -> List[Tuple[Document, float]]:
        """
        Run similarity search with Chroma with distance.
        """
        ret = self.vector_db.similarity_search_with_score(query=query, k=k)

    def similarity_search_with_relevance_scores(self, query: str, k: int = 4) -> List[Tuple[Document, float]]:
        """
        Return docs and relevance scores in the range [0, 1].
        0 is dissimilar, 1 is most similar.
        """
        return self.vector_db.similarity_search_with_relevance_scores(query=query, k=k)


vector_search = VectorSearch()
