"""
"""
import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.expanduser("~"), "src/langchain"))
from tqdm import tqdm
from typing import Dict, List

from langchain.chains import RetrievalQA
from langchain.document_loaders import UnstructuredFileLoader
from evaluator_app import (
    generate_eval as generate_example,
    split_texts,
    make_llm,
    make_retriever,
    make_chain,
    run_eval,
)


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def timestamp() -> str:
    from datetime import datetime
    from pytz import timezone

    return datetime.now(timezone("US/Pacific")).strftime("%Y-%m-%d-%H-%M")


def generate_qa_dataset(text: str, chunk_size: int, dataset_size: int):
    """Query LM to synthesize QA pairs from chunks of text."""
    dataset = []
    for _ in tqdm(range(dataset_size)):
        example = generate_example(text, chunk_size, logger)[0]
        dataset.append(example)
    return dataset


def generate_seq2seq_dataset(qa_dataset: List[Dict[str, str]], qa_chain: RetrievalQA):
    """Run retrieval step and generate prompt, completion pairs
    for fine-tuning QA step.
    """
    # https://github.com/hwchase17/langchain/commit/d1b92537b00db5a1eb09bcaa448652781b679b5a
    # I have to hack to expose the prompt that's been "stuffed" with the result of retrieval.
    qa_chain.combine_documents_chain.llm_chain.populate_prompt_cache = True
    qa_chain.combine_documents_chain.llm_chain.prompt_template_cache_key = "question"

    for example in tqdm(qa_dataset):
        qa_chain(example["question"])

    qa_chain.combine_documents_chain.llm_chain.populate_prompt_cache = (
        False  # Unset to un-shunt generations.
    )

    out = []
    for example in tqdm(qa_dataset):
        question = example["question"]
        raw_prompt = qa_chain.combine_documents_chain.llm_chain.prompt_cache[question]
        out.append({**example, "question_with_context": raw_prompt})
    return out
