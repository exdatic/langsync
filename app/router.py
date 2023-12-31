from datetime import datetime
import json
import os

from elasticsearch import Elasticsearch
from fastapi import APIRouter, File, HTTPException, UploadFile
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import ElasticsearchStore
from loguru import logger


router = APIRouter(tags=["API"])
es_url = os.environ.get("ES_URL", "http://elasticsearch:9200")


def publish_index(es: Elasticsearch, index_name: str, alias: str):
    old_index_names = es.indices.get_alias(index=alias, ignore_unavailable=True)
    if old_index_names:
        old_index_name = next(iter(old_index_names.keys()))
        es.indices.update_aliases(body={
            "actions": [
                {"remove": {"alias": alias, "index": old_index_name}},
                {"add": {"alias": alias, "index": index_name}},
            ]
        })
        logger.info(f"Updated alias {alias} to {index_name}")
        es.indices.delete(index=old_index_name)
        logger.info(f"Deleted old index {old_index_name}")
    else:
        es.indices.put_alias(index=index_name, name=alias)
        logger.info(f"Created alias {alias} for {index_name}")


@router.post("/sync", description="Update Elasticsearch index with data from JSON lines file")
def sync(
    index_name: str,
    jsonl_file: UploadFile = File(...),
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
    vector_query_field: str = "embedding",
    query_field: str = "text",
    timeout: int = 300
):
    texts = []
    metadatas = []
    for line in jsonl_file.file.readlines():
        doc = json.loads(line)
        if "content" not in doc:
            raise HTTPException(400, detail="JSON lines entry must contain 'content' field containing the text")
        texts.append(doc["content"])
        if "source" not in doc:
            raise HTTPException(400, detail="JSON lines entry must contain 'source' field containing the source url")
        if "title" not in doc:
            raise HTTPException(400, detail="JSON lines entry must contain 'title' field containing the title")
        metadatas.append({"source": doc["source"], "title": doc["title"]})

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, encoding_name="cl100k_base")
    chunks = text_splitter.create_documents(texts=texts, metadatas=metadatas)
    logger.info(f"Split into {len(chunks)} chunks")

    # temporal index name
    real_index_name = index_name + "-" + datetime.now().strftime("%Y%m%d%H%M%S%f")

    # fix https://github.com/langchain-ai/langchain/issues/4575
    if os.environ.get("OPENAI_API_TYPE") == "azure":
        embedding_chunk_size = 16
    else:
        embedding_chunk_size = 1000

    embedding = OpenAIEmbeddings(chunk_size=embedding_chunk_size, max_retries=100)  # type: ignore

    es = Elasticsearch(es_url, timeout=timeout)
    es_store = ElasticsearchStore(real_index_name,
                                  embedding=embedding,
                                  es_connection=es,
                                  vector_query_field=vector_query_field,
                                  query_field=query_field)
    es_store.add_documents(chunks)
    logger.info(f"Indexed {len(chunks)} chunks")
    publish_index(es, real_index_name, index_name)
    return "Updated data"
