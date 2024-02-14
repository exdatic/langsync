from datetime import datetime
import io
import json
import os
import re
from typing import Annotated

from datasets import Dataset
from elasticsearch import Elasticsearch
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from langchain.embeddings import CacheBackedEmbeddings
from langchain.schema import Document
from langchain.storage import LocalFileStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import ElasticsearchStore
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings
from loguru import logger
import pandas as pd
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, multi_context, reasoning

router = APIRouter(tags=["API"])
es_url = os.environ.get("ES_URL", "http://elasticsearch:9200")

if os.environ.get("OPENAI_API_TYPE") == "azure":
    embedding = AzureOpenAIEmbeddings(max_retries=100,
                                      azure_endpoint=os.environ.get("OPENAI_API_BASE",
                                                                    os.environ.get("AZURE_OPENAI_ENDPOINT")),
                                      validate_base_url=False)
else:
    embedding = OpenAIEmbeddings(max_retries=100)


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
def sync(
    index_name: str,
    jsonl_file: Annotated[list[UploadFile],
                          File(description="JSON lines file containing 'content', 'source' and 'title' fields")],
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
    vector_query_field: str = "embedding",
    query_field: str = "text",
    timeout: int = 300
):
    texts, metadatas = load_files(jsonl_file)

    if not texts:
        raise HTTPException(400, detail="No content found in JSON lines file")

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, encoding_name="cl100k_base")
    chunks = text_splitter.create_documents(texts=texts, metadatas=metadatas)
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
    es_store.add_documents(chunks)
    logger.info(f"Indexed {len(chunks)} chunks")
    publish_index(es, real_index_name, index_name)
    return "Updated data"


@router.post("/testset/create",
             description="Create a testset with data from JSON lines files",
             response_class=StreamingResponse)
def create_testset(
    jsonl_file: Annotated[list[UploadFile],
                          File(description="JSON lines file containing 'content', 'source' and 'title' fields")],
    test_size: int,
    distribution_simple: float = 0.75,
    distribution_multi: float = 0.25,
    distribution_reasoning: float = 0.0,
    generator_llm: str = "gpt-35-turbo",
    critic_llm: str = "gpt-4-turbo"
):
    texts, metadatas = load_files(jsonl_file)
    documents = []
    for text, metadata in zip(texts, metadatas):
        metadata["filename"] = metadata["source"]
        documents.append(Document(page_content=text, metadata=metadata))

    generator = TestsetGenerator.with_openai(generator_llm=generator_llm, critic_llm=critic_llm)
    testset = generator.generate_with_langchain_docs(documents, test_size, distributions={
        simple: distribution_simple, multi_context: distribution_multi, reasoning: distribution_reasoning})
    testset_df = testset.to_pandas()
    stream = io.StringIO(testset_df.to_csv(index=False))
    response = StreamingResponse(iter([stream.getvalue().encode()]), media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=testset.csv"
    return response


@router.post("/testset/evaluate", description="Evaluate a testset", response_model=dict)
def evaluate_testset(csv_file: UploadFile, eval_llm: str = "gpt-35-turbo"):
    logger.info("Reading testset")
    testset_df = pd.read_csv(io.StringIO(csv_file.file.read().decode()))
    if "question" not in testset_df.columns:
        raise HTTPException(400, detail="Testset must contain 'question' column")
    if "answer" not in testset_df.columns:
        raise HTTPException(400, detail="Testset must contain 'answer' column")
    if "contexts" not in testset_df.columns:
        raise HTTPException(400, detail="Testset must contain 'contexts' column")
    if "ground_truth" not in testset_df.columns:
        raise HTTPException(400, detail="Testset must contain 'ground_truth' column")
    # transform string to list
    testset_df.contexts = testset_df.contexts.apply(lambda s: s.replace("'", "\"")).apply(json.loads)

    # RAGAS test
    if os.environ.get("OPENAI_API_TYPE") == "azure":
        llm = AzureChatOpenAI(
            azure_deployment=eval_llm,
            temperature=0.0,
            max_tokens=1000,
            streaming=False
        )
    else:
        llm = ChatOpenAI(
            model=eval_llm,
            temperature=0.0,
            max_tokens=1000,
            streaming=False
        )
    dataset = Dataset.from_pandas(testset_df[["question", "answer", "contexts", "ground_truth"]])
    llm_wrapper = LangchainLLMWrapper(langchain_llm=llm)
    emb_wrapper = LangchainEmbeddingsWrapper(embeddings=embedding)
    logger.info("Evaluating testset with RAGAS")
    evaluation = evaluate(
        dataset,
        metrics=[
            answer_relevancy,
            faithfulness,
            context_recall,
            context_precision,
        ],
        llm=llm_wrapper,
        embeddings=emb_wrapper
    )
    result_df: pd.DataFrame = evaluation.to_pandas()  # type: ignore
    eval_dict = {k: round(float(v), 2) for k, v in evaluation.items()}
    # test correctness of sources
    if "filenames" in testset_df.columns:
        # transform string to list
        testset_df.filenames = testset_df.filenames.apply(lambda s: s.replace("'", "\"")).apply(json.loads)

        url_pattern = r"(https?://[^\s()]+)"
        testset_df['sources'] = testset_df['answer'].apply(lambda a: re.findall(url_pattern, a))
        result_df['correct_sources'] = testset_df.apply(
            lambda row: all(url in row['sources'] for url in row['filenames']), axis=1
        )
        result_df['sources'] = testset_df['sources']
        eval_dict['source_accuracy'] = float(result_df['correct_sources'].mean())
    return {
        "results": json.loads(result_df.to_json(orient='records')),
        "evaluation": eval_dict
    }


def load_files(jsonl_file):
    texts = []
    metadatas = []
    for upload_file in jsonl_file:
        for line in upload_file.file.readlines():
            doc = json.loads(line)
            if "content" not in doc:
                raise HTTPException(400, detail="JSON lines entry must contain 'content' field containing the text")
            if "source" not in doc:
                raise HTTPException(400, detail="JSON lines entry must contain 'source' field containing the source url")
            if "title" not in doc:
                raise HTTPException(400, detail="JSON lines entry must contain 'title' field containing the title")
            texts.append("\n\n".join([doc["title"], doc["content"]]))
            metadata = {k: v for k, v in doc.items() if k not in ["content"]}
            metadata["filename"] = upload_file.filename
            metadatas.append(metadata)
    return texts, metadatas
