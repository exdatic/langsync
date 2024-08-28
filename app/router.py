from datetime import datetime
import json
import os
import time
from typing import Annotated

from elasticsearch import Elasticsearch
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_elasticsearch import ElasticsearchStore
from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings
from loguru import logger


router = APIRouter(tags=["API"])
es_url = os.environ.get("ES_URL", "http://elasticsearch:9200")


def publish_index(es: Elasticsearch, index_name: str, alias: str):
    old_index_names = es.indices.get_alias(index=alias, ignore_unavailable=True)
    if old_index_names:
        old_index_name = next(iter(old_index_names.keys()))
        es.indices.update_aliases(actions=[
            {"remove": {"alias": alias, "index": old_index_name}},
            {"add": {"alias": alias, "index": index_name}},
        ])
        logger.info(f"Updated alias {alias} to {index_name}")
        es.indices.delete(index=old_index_name)
        logger.info(f"Deleted old index {old_index_name}")
    else:
        es.indices.put_alias(index=index_name, name=alias)
        logger.info(f"Created alias {alias} for {index_name}")


@router.post("/sync", description="Update Elasticsearch index with data from JSON lines file")
async def sync(
    index_name: str,
    jsonl_file: Annotated[list[UploadFile],
                          File(description="JSON lines file containing 'content', 'source' and 'title' fields")],
    background_tasks: BackgroundTasks,
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
    batch_size: int = 256,
    add_titles_to_chunks: bool = True,
    vector_query_field: str = "embedding",
    query_field: str = "text",
    timeout: int = 300
):
    texts, metadatas = load_files(jsonl_file)

    if not texts:
        raise HTTPException(400, detail="No content found in JSON lines file")
    
    if os.environ.get("OPENAI_API_TYPE") == "azure":
        embedding = AzureOpenAIEmbeddings(model='text-embedding-3-large', max_retries=100, chunk_size=batch_size)
    else:
        embedding = OpenAIEmbeddings(max_retries=100)

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n\n", "\n\n", "\n", " ", ""],
        encoding_name="cl100k_base")

    chunks = text_splitter.create_documents(texts=texts, metadatas=metadatas)
    if add_titles_to_chunks:
        for c in chunks:
            c.page_content = c.metadata['title'] + "\n\n" + c.page_content

    logger.info(f"Split into {len(chunks)} chunks")

    # temporal index name
    real_index_name = index_name + "-" + datetime.now().strftime("%Y%m%d%H%M%S%f")

    fs = LocalFileStore("/cache/")
    cached_embedder = CacheBackedEmbeddings.from_bytes_store(embedding, fs, namespace=index_name)
    es = Elasticsearch(es_url, timeout=timeout)
    es_store = ElasticsearchStore(real_index_name,
                                  embedding=cached_embedder,
                                  es_connection=es,
                                  vector_query_field=vector_query_field,
                                  query_field=query_field)

    logger.info(f"Indexing {len(chunks)} text chunks in batches of {batch_size}")

    async def index_chunks(chunks):
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            try:
                await es_store.aadd_documents(batch)
            except Exception as e:
                logger.error(f"Failed to index batch {i}-{i + batch_size}: {e}")
                logger.info("Sleeping for 1 minute")
                time.sleep(60)
            logger.info(f"Indexed {len(batch)} text chunks")
        logger.info(f"Indexed all {len(chunks)} chunks")
        publish_index(es, real_index_name, index_name)

    background_tasks.add_task(index_chunks, chunks)
    return f"Updating {len(chunks)} text chunks in background"


def load_files(jsonl_file):
    texts = []
    metadatas = []
    for upload_file in jsonl_file:
        for line in upload_file.file.readlines():
            doc = json.loads(line)
            if "content" not in doc:
                raise HTTPException(400, detail="JSON lines entry must contain 'content' field containing the text")
            if "source" not in doc:
                raise HTTPException(400, detail="JSON lines entry must contain 'source' field containing the url")
            if "title" not in doc:
                raise HTTPException(400, detail="JSON lines entry must contain 'title' field containing the title")
            texts.append(doc["content"])
            metadata = {k: v for k, v in doc.items() if k not in ["content"]}
            metadata["filename"] = upload_file.filename
            metadatas.append(metadata)
    return texts, metadatas
