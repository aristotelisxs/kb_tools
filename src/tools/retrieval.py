import asyncio

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

from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    TypeVar,
    Union,
)

from langchain_core.runnables.utils import (
    ConfigurableFieldSpec,
    get_unique_config_specs,
)
from pydantic import model_validator
from langchain_core.load.dump import dumpd
from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.runnables import RunnableConfig
from collections import defaultdict
from itertools import chain
from langchain_core.runnables.config import ensure_config, patch_config

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None

from langchain_core.retrievers import BaseRetriever, RetrieverLike


T = TypeVar("T")
H = TypeVar("H", bound=Hashable)


def unique_by_key(iterable: Iterable[T], key: Callable[[T], H]) -> Iterator[T]:
    """Yield unique elements of an iterable based on a key function.

    Args:
        iterable: The iterable to filter.
        key: A function that returns a hashable key for each element.

    Yields:
        Unique elements of the iterable based on the key function.
    """
    seen = set()
    for e in iterable:
        if (k := key(e)) not in seen:
            seen.add(k)
            yield e

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


class ReRankEnsembleRetriever(BaseRetriever):
    """Retriever that ensembles the multiple retrievers.

    It uses a rank fusion.

    Args:
        retrievers: A list of retrievers to ensemble.
        weights: A list of weights corresponding to the retrievers. Defaults to equal
            weighting for all retrievers.
        c: A constant added to the rank, controlling the balance between the importance
            of high-ranked items and the consideration given to lower-ranked items.
            Default is 60.
        id_key: The key in the document's metadata used to determine unique documents.
            If not specified, page_content is used.
    """

    retrievers: List[RetrieverLike]
    id_key: Optional[str] = None
    cross_encoder: CrossEncoder = (
        CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        if CrossEncoder and cfg.retriever.use_semantic_scorer
        else None
    )
    at_least_docs: int = cfg.retriever.min_docs
    weights: List[float]
    c: int = 60
    use_wgv_over_top_k: bool

    @property
    def config_specs(self) -> List[ConfigurableFieldSpec]:
        """List configurable fields for this runnable."""
        return get_unique_config_specs(
            spec for retriever in self.retrievers for spec in retriever.config_specs
        )

    @model_validator(mode="before")
    def set_weights(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if not values.get("weights"):
            n_retrievers = len(values["retrievers"])
            values["weights"] = [1 / n_retrievers] * n_retrievers
        return values

    def invoke(
        self, input: str, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> List[Document]:
        from langchain_core.callbacks import CallbackManager

        config = ensure_config(config)
        callback_manager = CallbackManager.configure(
            config.get("callbacks"),
            None,
            verbose=kwargs.get("verbose", False),
            inheritable_tags=config.get("tags", []),
            local_tags=self.tags,
            inheritable_metadata=config.get("metadata", {}),
            local_metadata=self.metadata,
        )
        run_manager = callback_manager.on_retriever_start(
            dumpd(self),
            input,
            name=config.get("run_name"),
            **kwargs,
        )
        try:
            result = self.rank_fusion(input, run_manager=run_manager, config=config)
        except Exception as e:
            run_manager.on_retriever_error(e)
            raise e
        else:
            run_manager.on_retriever_end(
                result,
                **kwargs,
            )
            return result

    async def ainvoke(
        self, input: str, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> List[Document]:
        from langchain_core.callbacks import AsyncCallbackManager

        config = ensure_config(config)
        callback_manager = AsyncCallbackManager.configure(
            config.get("callbacks"),
            None,
            verbose=kwargs.get("verbose", False),
            inheritable_tags=config.get("tags", []),
            local_tags=self.tags,
            inheritable_metadata=config.get("metadata", {}),
            local_metadata=self.metadata,
        )
        run_manager = await callback_manager.on_retriever_start(
            dumpd(self),
            input,
            name=config.get("run_name"),
            **kwargs,
        )
        try:
            result = await self.arank_fusion(
                input, run_manager=run_manager, config=config
            )
        except Exception as e:
            await run_manager.on_retriever_error(e)
            raise e
        else:
            await run_manager.on_retriever_end(
                result,
                **kwargs,
            )
            return result

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """
        Get the relevant documents for a given query.

        Args:
            query: The query to search for.

        Returns:
            A list of reranked documents.
        """

        # Get fused result of the retrievers.
        fused_documents = self.rank_fusion(query, run_manager)

        return fused_documents

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """
        Asynchronously get the relevant documents for a given query.

        Args:
            query: The query to search for.

        Returns:
            A list of reranked documents.
        """

        # Get fused result of the retrievers.
        fused_documents = await self.arank_fusion(query, run_manager)

        return fused_documents

    def rank_fusion(
        self,
        query: Union[str, dict],
        run_manager: CallbackManagerForRetrieverRun,
        *,
        config: Optional[RunnableConfig] = None,
    ) -> List[Document]:
        """
        Synchronously retrieve the results of the retrievers
        and use rank_fusion_func to get the final result.

        Returns:
            A list of reranked documents.
        """

        retriever_docs = [
            retriever.invoke(
                query,
                patch_config(
                    config, callbacks=run_manager.get_child(tag=f"retriever_{i + 1}")
                ),
            )
            for i, retriever in enumerate(self.retrievers)
        ]

        return self._get_fused_documents(query, retriever_docs)

    async def arank_fusion(
        self,
        query: str,
        run_manager: AsyncCallbackManagerForRetrieverRun,
        *,
        config: Optional[RunnableConfig] = None,
    ) -> List[Document]:
        """
        Asynchronously retrieve the results of the retrievers
        and use rank_fusion_func to get the final result.

        Args:
            query: The query to search for.

        Returns:
            A list of reranked documents.
        """

        # Get the results of all retrievers.
        retriever_docs = await asyncio.gather(
            *[
                retriever.ainvoke(
                    query,
                    patch_config(
                        config,
                        callbacks=run_manager.get_child(tag=f"retriever_{i + 1}"),
                    ),
                )
                for i, retriever in enumerate(self.retrievers)
            ]
        )

        return self._get_fused_documents(query, retriever_docs)

    def _get_fused_documents(
        self, query: Union[str, dict], documents: List[List[Document]]
    ) -> List[Document]:
        for i in range(len(documents)):
            documents[i] = [
                Document(page_content=doc) if not isinstance(doc, Document) else doc  # type: ignore[arg-type]
                for doc in documents[i]
            ]

        fused_documents = self.weighted_reciprocal_rank(documents)

        logger.debug(
            f"Used rank fusion on retrieved documents, New scores are: {[[doc.metadata['score'], doc.metadata['search']] for doc in fused_documents]}"
        )

        if self.cross_encoder:
            fused_documents = self.cross_encoder_re_ranking(
                query, fused_documents
            )

        if self.use_wgv_over_top_k:
            fused_documents = filter_documents(fused_documents, self.at_least_docs)

        return fused_documents

    def cross_encoder_re_ranking(
        self, query: str, doc_list: List[Document]
    ) -> List[Document]:
        """
        Perform Retrieve & Re-Rank on multiple rank lists.
        You can find more details about RRF here:
        https://github.com/UKPLab/sentence-transformers/blob/master/examples/applications/retrieve_rerank/retrieve_rerank_simple_wikipedia.ipynb

        Args:
            doc_list: A ranked list, where each element contains a unique document.

        Returns:
            list: The final aggregated list of items sorted by their cross_encoder re-ranked
                    scores in descending order.
        """
        cross_inp = [[query, doc.page_content] for doc in doc_list]
        cross_scores = self.cross_encoder.predict(cross_inp)

        for rank, doc in enumerate(doc_list):
            doc.metadata["score"] = cross_scores[rank]

        # Docs are deduplicated by their contents then sorted by their scores
        sorted_docs = sorted(
            doc_list, reverse=True, key=lambda doc: doc.metadata["score"]
        )

        return sorted_docs

    def weighted_reciprocal_rank(
        self, doc_lists: List[List[Document]]
    ) -> List[Document]:
        """
        Perform weighted Reciprocal Rank Fusion on multiple rank lists.
        You can find more details about RRF here:
        https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf

        Args:
            doc_lists: A list of rank lists, where each rank list contains unique items.

        Returns:
            list: The final aggregated list of items sorted by their weighted RRF
                    scores in descending order.
        """
        if len(doc_lists) != len(self.weights):
            raise ValueError(
                "Number of rank lists must be equal to the number of weights."
            )

        # Associate each doc's content with its RRF score for later sorting by it
        # Duplicated contents across retrievers are collapsed & scored cumulatively

        logger.debug(f"Got {sum(map(len, doc_lists))} documents in total before WRRF")

        rrf_score: dict[str, float] = defaultdict(float)
        searches: dict[str, list] = defaultdict(list)
        for doc_list, weight in zip(doc_lists, self.weights):
            unique_doc_ids = defaultdict(int)
            for rank, doc in enumerate(doc_list, start=1):
                i = doc.id if doc.id is not None else doc.page_content
                rrf_score[i] += weight / (rank + self.c)
                searches[i].append(doc.metadata["search"])

                unique_doc_ids[i] += 1

            logger.debug(
                f"Total unique documents: {len(unique_doc_ids)} they are: {unique_doc_ids}"
            )

        all_docs = chain.from_iterable(doc_lists)
        sorted_docs = sorted(
            unique_by_key(
                all_docs,
                lambda doc: (doc.id if doc.id is not None else doc.page_content),
            ),
            reverse=True,
            key=lambda doc: rrf_score[
                doc.id if doc.id is not None else doc.page_content
            ],
        )

        for doc in sorted_docs:
            i = doc.id if doc.id is not None else doc.page_content
            doc.metadata["score"] = rrf_score[i]
            doc.metadata["search"] = searches[i]

        logger.debug(
            f"Got {len(sorted_docs)} documents after WRRF with {[doc.metadata['search'] for doc in sorted_docs]} searches"
        )

        return sorted_docs


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
