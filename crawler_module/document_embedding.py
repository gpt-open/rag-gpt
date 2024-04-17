# coding=utf-8
import json
import time
from langchain.schema.document import Document
from langchain_community.vectorstores import Chroma
from utils.logger_config import my_logger as logger


class DocumentEmbedder:

    def __init__(self, collection_name, embedding_function, persist_directory):
        logger.info(f"[DOC_EMBEDDING] init, collection_name:'{collection_name}', persist_directory:{persist_directory}")
        collection_metadata = {"hnsw:space": "cosine"}
        self.chroma_obj = Chroma(
                collection_name=collection_name,
                embedding_function=embedding_function,
                persist_directory=persist_directory,
                collection_metadata=collection_metadata)

    async def aadd_content_embedding(self, data):
        records_to_add = []
        records_to_update = []
        for item in data:
            documents_to_add = []  # Initialize a list to hold all document parts for batch processing.
            timestamp = int(time.time())
            doc_id, url, chunk_text_vec = item
            part_index = 0
            for part_content in chunk_text_vec:
                # Process each part of the content.
                # Construct metadata for each document part, ensuring each part has a unique ID.
                metadata = {"source": url, "id": f"{doc_id}-part{part_index}"}
                # Create a Document object with the part content and metadata.
                doc = Document(page_content=part_content, metadata=metadata)
                # Add the document part to the list for batch addition.
                documents_to_add.append(doc)
                part_index += 1

            # Check if there are document parts to add.
            if documents_to_add:
                # Add all document parts to Chroma in a single batch operation.
                embedding_id_vec = await self.chroma_obj.aadd_documents(documents_to_add)
                logger.info(f"[DOC_EMBEDDING] doc_id={doc_id}, url={url} added {len(documents_to_add)} document parts to Chroma., embedding_id_vec={embedding_id_vec}")
                records_to_add.append((doc_id, json.dumps(embedding_id_vec), timestamp, timestamp))
                records_to_update.append((4, timestamp, doc_id))

        return records_to_add, records_to_update

    #async def adelete_content_embedding(self, embedding_id_vec):
    #    return await self.chroma_obj.adelete(embedding_id_vec)

    def delete_content_embedding(self, embedding_id_vec):
        return self.chroma_obj.delete(embedding_id_vec)

    def search_document(self, query, k):
        return self.chroma_obj.similarity_search_with_score(query, k=k)
