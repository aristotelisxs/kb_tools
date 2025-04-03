from collections.abc import Hashable
from typing import (
    Any,
    TypeVar,
)

import numpy as np
from langchain.vectorstores.base import VectorStoreRetriever
from opensearchpy import NotFoundError
from scipy.optimize import dual_annealing, minimize
from langchain_core.documents import Document

from tools.utils import get_opensearch_client
from tools.config import cfg
from tools.log import logger

T = TypeVar("T")
H = TypeVar("H", bound=Hashable)


def within_group_variance(threshold, data):
    group1 = data[data <= threshold]
    group2 = data[data > threshold]
    variance1 = np.var(group1) if len(group1) > 0 else 0
    variance2 = np.var(group2) if len(group2) > 0 else 0
    return variance1 + variance2


def find_score_cutoff(scores, min_docs):
    # TODO: Repeat until minimum requirement of documents reached.
    def objective(threshold, data):
        return within_group_variance(threshold, np.array(data))

    doc_count = 0
    optimal_threshold = None
    # Enforce document count
    while doc_count < min_docs:
        if min(scores) >= max(scores):
            return optimal_threshold

        bounds = [(min(scores), max(scores))]

        # Obtain global minima as initial guess
        result_sa = dual_annealing(objective, bounds, args=(scores,))
        global_threshold = result_sa.x[0]

        # Refine using Nelder-Mead
        result_local = minimize(
            objective, global_threshold, args=(scores,), method="Nelder-Mead"
        )
        optimal_threshold = result_local.x[0]

        filt_scores = [s for s in scores if s >= optimal_threshold]
        doc_count = len(filt_scores)
        # Remove points that influenced cut-off selection and try again if minimum documents not reached..
        scores = [s for s in scores if s < optimal_threshold]

    return optimal_threshold


def filter_documents(doc_list: list[Document], min_docs: int | None):
    if min_docs is None:
        return doc_list
    # assuming everything is sorted here
    # Initial guess for the threshold
    scores = [doc.metadata["score"] for doc in doc_list]

    optimal_threshold = None
    # Optimize the threshold. If the document set is too small already, return as is.
    if len(scores) > min_docs:
        optimal_threshold = find_score_cutoff(scores, min_docs)
        filtered_scores = [
            s
            for s in scores
            if optimal_threshold is not None and s >= optimal_threshold
        ]
    else:
        filtered_scores = scores
    n_filtered = len(filtered_scores)

    logger.debug(
        f"Initial scores were {scores}. Selected {n_filtered} documents with {optimal_threshold} score threshold"
    )
    if n_filtered < min_docs:
        logger.debug(
            f"Returning {min_docs} because {n_filtered} filtered docs are too small"
        )

    return doc_list[: max(n_filtered, min_docs)]


def get_documents_by_id(os_index: str, ids: list[str], _source: list[str]):
    os_client = get_opensearch_client()

    body = {
        "docs": [
            {"_index": os_index, "_id": doc_id, "_source": _source} for doc_id in ids
        ]
    }

    response = os_client.mget(body=body)
    os_client.close()

    return response


class VectorStoreRetrieverWithScores(VectorStoreRetriever):
    """VectorStoreRetriever that returns scores with metadata in case of
    'similarity_score_threshold' search type
    """

    use_wgv_over_top_k: bool = False
    at_least_docs: int | None = cfg.retriever.min_docs
    handle_missing_index: bool = False

    def get_relevant_documents(self, query: str) -> list[Document]:
        try:
            if self.search_type == "similarity_score_threshold":
                docs_and_similarities = (
                    self.vectorstore.similarity_search_with_relevance_scores(
                        query, **self.search_kwargs
                    )
                )

                docs = []
                for doc, similarity in docs_and_similarities:
                    doc.metadata["score"] = similarity
                    doc.metadata["search"] = "vector"
                    docs.append(doc)

            else:
                docs = super(
                    VectorStoreRetrieverWithScores, self
                ).get_relevant_documents(query, **self.search_kwargs)
        except NotFoundError:
            if self.handle_missing_index:
                logger.warning("Index not found. Returning empty list")
                return []
            raise

        if self.use_wgv_over_top_k:
            docs = filter_documents(docs, self.at_least_docs)

        return docs

    async def aget_relevant_documents(self, query: str) -> list[Document]:
        try:
            if self.search_type == "similarity_score_threshold":
                docs_and_similarities = (
                    await self.vectorstore.asimilarity_search_with_relevance_scores(
                        query, **self.search_kwargs
                    )
                )

                docs = []
                for doc, similarity in docs_and_similarities:
                    doc.metadata["score"] = similarity
                    doc.metadata["search"] = "vector"
                    docs.append(doc)

            else:
                docs = await super(
                    VectorStoreRetrieverWithScores, self
                ).aget_relevant_documents(query, **self.search_kwargs)
        except NotFoundError:
            if self.handle_missing_index:
                logger.warning("Index not found. Returning empty list")
                return []
            raise

        logger.debug(f"Received {len(docs)} from vector search")

        if self.use_wgv_over_top_k:
            docs = filter_documents(docs, self.at_least_docs)

        return docs
