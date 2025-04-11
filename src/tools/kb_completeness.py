import json
import re
import time
import traceback
from collections import defaultdict
from copy import copy
from typing import Dict, List, Union

import nltk
import botocore
import numpy as np
import typer
from langchain.schema.output_parser import StrOutputParser
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from nltk import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from tools.utils import (
    combine_docs,
    get_bedrock_client,
    get_prompt,
    get_vector_store,
)
from tools.config import cfg
from tools.log import logger
from tools.prompt_correcter import Correcter
from tools.retrieval import (
    VectorStoreRetrieverWithScores,
    get_documents_by_id,
)

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None


def strip_agent_answers(sentences: List[str]) -> List[str]:
    """Agent replies to customers include opening and closing remarks.
    This is a hard-coded function to remove the first and last 2 sentences that correspond to each of these remarks.

    Args:
        sentences (List[str]): List of each sentence in the agent's reply

    Returns:
        List[str]: The filtered from remarks list of sentences
    """

    # Remove the first and last sentence, then join the rest
    start_pos = 2
    end_pos = -2

    if len(sentences) > 2:
        if sentences[start_pos].startswith("We appreciate"):
            start_pos += 1

        return sentences[start_pos:end_pos]
    else:
        return sentences


def count_sentences(text: str) -> int:
    """Count the number of sentences in text, ending with the following punctuations: ".", "?", "!"

    Args:
        text (str): The text to count sentences from.

    Returns:
        int: The number of sentences in text.
    """

    # Regular expression pattern to split on sentence-ending punctuation (., ?, !)
    sentences = re.split(r"[.!?]", text)

    # Remove any empty sentences that may result from trailing punctuation
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

    return len(sentences)


def check_float_in_range(
    s: str | float | int, min_val: float = 0.0, max_val: float = 1.0
) -> None | float:
    """Check if the given number or numeric string is within specified bounds.

    Args:
        s (str | float | int):
        min_val (float, optional): Lower bound. Defaults to 0..
        max_val (float, optional): Upper bound. Defaults to 1..

    Returns:
        None | float: None if not within bounds, else the number itself.
    """
    try:
        # Try to convert the string to a float
        value = float(s)

        # Check if the value is between 0 and 1
        if min_val <= value <= max_val:
            return value
    except ValueError:
        pass

    return None


def invoke_correcter(question: str) -> str:
    correcting_prompt = get_prompt(
        cfg.prompt_preproc.correcting_prompt_name, return_str=True
    )

    chat_prompt = ChatPromptTemplate(
        [
            (
                cfg.common_prompt_tags.system,
                correcting_prompt,
            ),
            (cfg.common_prompt_tags.human, "{input}"),
        ]
    )

    correcter = Correcter(
        client=get_bedrock_client(),
        model_id=cfg.bedrock.correcting_model.bedrock_model_id,
        max_tokens=cfg.bedrock.correcting_model.max_tokens,
        temperature=cfg.bedrock.correcting_model.temperature,
        model_kwargs=cfg.bedrock.correcting_model.get_model_kwargs(),
    )

    chain = (chat_prompt | correcter)

    reply = chain.invoke(
        {"input": question}
    )

    pattern = r"<text>(.*?)</text>"

    # Extract all matches
    matches = re.findall(pattern, reply, re.DOTALL)

    if len(matches) == 1:
        # Return the first match. There should only be one
        return matches[0]
    else:
        # Otherwise, return the original question
        return question
        

class KBEvaluator:
    def __init__(
        self,
        docs_to_retrieve: int = cfg.kb_completeness.docs_to_retrieve,
        docs_in_prompt: int = cfg.kb_completeness.docs_in_prompt,
        cluster_size: int = cfg.kb_completeness.cluster_size,
        excerpt_score_threshold: float = cfg.kb_completeness.excerpt_score_threshold,
        use_reranking: bool = cfg.kb_completeness.use_reranking,
        evaluate_at: int = cfg.kb_completeness.evaluate_at,
        improvement_by: float = cfg.kb_completeness.improvement_by,
        stop_if_no_improvement_for: int = cfg.kb_completeness.stop_if_no_improvement_for,
    ):
        """_summary_

        Args:
            docs_to_retrieve (int, optional): Maximum number of documents to retrieve from KB for each question. Defaults to 100.
            docs_in_prompt (int, optional): Maximum number of documents to use in each LLM prompt for doc-answer excerpts mapping extraction. Defaults to 15.
            excerpt_score_threshold (int, optional): Reject doc-answer excerpt mapping if below this threshold. Give a score from 0-1. Defaults to 0.5.
            use_reranking (bool, optional): Whether to use the . Defaults to True.
            evaluate_at (int, optional): At what step during page iteration to start checking if there are any improvements to score. Defaults to 3.
            improvement_by (float, optional): The % of improvement that needs to occur for iterations to continue. Defaults to 5.
            stop_if_no_improvement_for (int, optional): How many steps to tolerate no improvements until halting experiment for Q-A pair. Defaults to 3.
        """
        nltk.download("stopwords")

        self.cross_encoder = (
            CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            if CrossEncoder and use_reranking
            else None
        )

        self._lemmatizer = WordNetLemmatizer()
        self._stop_words = set(stopwords.words("english"))
        self.page_size = docs_in_prompt
        self.cluster_size = cluster_size
        self.score_threshold = excerpt_score_threshold

        search_type = "similarity_score_threshold"
        search_kwargs = {
            "k": docs_to_retrieve
            if cfg.kb_completeness.reranking_mode == "answer_based"
            else docs_to_retrieve / 2,
            "score_threshold": 0.0,  # Documents that have at least this similarity score. Allow all documents that demonstrate even the slightest similarity
        }
        self.retriever = VectorStoreRetrieverWithScores(
            vectorstore=get_vector_store(None),
            search_type=search_type,
            search_kwargs=search_kwargs,
            use_wgv_over_top_k=False,  # We are performing a KB-wide completeness check, so enforce maximum limit by user.
        )

        self.evaluate_at = evaluate_at
        self.improvement_by = improvement_by
        self.stop_if_no_improvement_for = stop_if_no_improvement_for

        self.bedrock_client = get_bedrock_client()
        self.jaccard_similarity_threshold = (
            cfg.kb_completeness.completeness_score_threshold
        )

        self.llm = ChatBedrock(
            client=get_bedrock_client(),
            model_id=cfg.kb_completeness.model_id,
            max_tokens=cfg.kb_completeness.max_tokens,
            temperature=cfg.kb_completeness.temperature,
        )

        self._embed_model = cfg.kb_completeness.query_embed_model
        self.correct_prompt = cfg.prompt_preproc.correct_prompt

        # Load model for embedding
        if "amazon.titan-embed" in self._embed_model:
            self._embed_model_obj = BedrockEmbeddings(
                client=get_bedrock_client(),
                model_id=self._embed_model,
            )
        else:
            raise ValueError(f"Invalid embed model: {self._embed_model}")

    def _get_embedding(self, text: str) -> np.ndarray | None:
        try:
            if isinstance(self._embed_model_obj, BedrockEmbeddings):
                embedding = self._embed_model_obj.embed_documents(texts=[text])[0]
                return np.asarray(embedding)
            else:
                logger.error(f"Unsupported model type: {type(self._embed_model_obj)}")
                return None
        except Exception as e:
            logger.warning(
                f"Failed to generate embedding for text: {text[:30]}... Error: {str(e)}"
            )
            return None    

    def llm_document_analysis(
        self,
        system_prompt: str,
        human_prompt: str,
        question: str,
        answer: str,
        doc_batch: List[Document],
    ) -> List[str]:
        # Prepare the prompt
        cur_docs = combine_docs(doc_batch, include_summary=False)
        chat_prompt = ChatPromptTemplate(
            [
                (
                    cfg.common_prompt_tags.system,
                    system_prompt,
                ),
                (cfg.common_prompt_tags.human, human_prompt),
            ]
        )

        chain = (chat_prompt | self.llm | StrOutputParser())

        reply = chain.invoke(
            {"question": question, "documents": cur_docs, "answer": answer}
        )

        pattern = r"<item>(.*?)</item>"

        # Extract all matches
        matches = re.findall(pattern, reply, re.DOTALL)

        return matches

    def extract_mapping_details(
        self,
        mappings: List[str],
        global_mappings: Dict[str, Dict[str, Union[List[str], List[float], str]]],
    ) -> Dict[str, Dict[str, Union[List[str], List[float], str]]]:
        """Extract excerpt information from the LLM reply within the KB eval process. Specifically, we are looking for items
        within this structured text:

        <item>
         <answer_excerpt></answer_excerpt>
         <document_excerpt></document_excerpt>
         <question_excerpt></question_excerpt>
         <score></score>
        </item>

        Args:
            mappings (List[str]): _description_
            global_mappings (Dict[str, Any]): _description_
        """
        score_tag = r"<score>(.*?)</score>"
        answer_tag = r"<answer_excerpt>(.*?)</answer_excerpt>"
        question_tag = r"<question_excerpt>(.*?)</question_excerpt>"
        doc_tag = r"<document_excerpt>(.*?)</document_excerpt>"

        for match in mappings:
            score = re.findall(score_tag, match, re.DOTALL)
            answer = re.findall(answer_tag, match, re.DOTALL)
            question = re.findall(question_tag, match, re.DOTALL)
            doc = re.findall(doc_tag, match, re.DOTALL)

            # There should only be a single match for each <item> for the above. Confirm.
            if len(score) != 1 or len(answer) != 1 or len(doc) != 1:
                # Malformed item structure. Skip
                continue

            score = check_float_in_range(score[0])
            if score is None or np.isnan(score):
                continue

            answer = answer[0]
            doc = doc[0]
            question = question[0]

            try:
                # Check if it already exists before appending (exact match)
                matched_answer = None
                matched_score = None
                for s, info in global_mappings.items():
                    # If there is any overlap, consider as similar, given that they also have content from the same sentences.
                    if (answer in s or s in answer) and (
                        count_sentences(answer) == count_sentences(s)
                    ):
                        matched_answer = s
                        matched_score = np.max(info["score"])
                        break

                if matched_answer is None:
                    # No similar excerpts found. Simply add.
                    global_mappings[answer] = {
                        "document": [doc],
                        "score": [score],
                        "question": question,
                    }
                else:
                    if score > matched_score:
                        global_mappings[matched_answer]["document"] += [doc]
                        global_mappings[matched_answer]["score"] += [score]
            except ValueError:
                # Reject because of bad score output
                pass

        return global_mappings

    @staticmethod
    def merge_docs_from_retrievers(results_1, results_2) -> List[Document]:
        # Set to keep track of documents already processed
        document_ids_1 = {doc.id: doc for doc in results_1}
        document_ids_2 = {doc.id: doc for doc in results_2}

        # Step 1: Identify overlap and average scores
        merged_results = []

        # Process overlapping documents
        for doc_id, doc_1 in document_ids_1.items():
            if doc_id in document_ids_2:
                doc_2 = document_ids_2[doc_id]
                # Average the scores of the overlapping documents
                avg_score = (doc_1.metadata["score"] + doc_2.metadata["score"]) / 2
                # Add the averaged document to merged results
                doc_1.metadata["score"] = avg_score
                merged_results.append(doc_1)
                # Mark doc_id as processed so we don't include it twice
                del document_ids_2[doc_id]

        # Step 2: Include non-overlapping documents
        merged_results.extend(
            list(document_ids_1.values())
        )  # Add non-overlapping from results_1
        merged_results.extend(
            list(document_ids_2.values())
        )  # Add non-overlapping from results_2

        # Optionally, you could sort these results by score
        merged_results = sorted(
            merged_results, key=lambda x: x.metadata["score"], reverse=True
        )

        return merged_results

    @staticmethod
    def remove_low_sim_docs(
        docs: List[Document], std_thresh: int = 2
    ) -> List[Document]:
        """Remove documents that are at least `std_thresh` standard deviations away

        Args:
            docs (List[Document]): List of Document objects to retrieve similarity scores from

        Returns:
            List[Document]: Remaining Document objects, after applying threshold
        """

        # Filter out outliers based on new score standard deviation.
        ce_scores = [np.max(x.metadata["ce_score"]) for x in docs]
        # TODO: Threshold subject to change after experimentation
        score_mean = round(np.mean(ce_scores), 2)
        score_std = round(np.std(ce_scores), 2)
        score_thresh = round(score_mean - (score_std * std_thresh), 2)

        relevant_docs = [
            x for x in docs if np.max(x.metadata["ce_score"]) > score_thresh
        ]

        logger.info(
            f"Filtered {len(relevant_docs) - len(docs)} documents due to extremely low relevance score. Score threshold: {score_thresh}, mean: {score_mean}, std: {score_std}"
        )

        return relevant_docs

    def evaluate_w_question(
        self, question: str, agent_answer: str
    ) -> Dict[str, Dict[str, Union[List[str], List[float], str]]]:
        """Calculate KB completeness score for specified question and answer pair.

        Args:
            question (str): Block of text containing player's statements/questions to check whether any information in KB can be used to validate/answer them.
            agent_answer (str): If there is an agent reply to the customer's question, provide it here. Statements will be used for document re-ranking and filtering to speed up the process. Defaults to None.
        """
        relevant_docs = self.retriever.get_relevant_documents(
            {"original": question, "lang": "en"}
        )

        if cfg.kb_completeness.reranking_mode == "document_based":
            extra_relevant_docs = self.retriever.get_relevant_documents(
                {"original": agent_answer, "lang": "en"}
            )

            relevant_docs = self.merge_docs_from_retrievers(
                relevant_docs, extra_relevant_docs
            )

        len_relevant_docs = len(relevant_docs)
        logger.info(
            f"Got {len_relevant_docs} documents from KB."
        )

        reranked_docs = relevant_docs
        # Re-rank based on answer. Cannot use both because of information duplication that might bias scoring.
        statements = sent_tokenize(agent_answer)
        statements = strip_agent_answers(statements)
        agent_answer = "\n".join(statements)

        reranked_docs = self.doc_reranking(docs=relevant_docs, answer=statements)
        if cfg.kb_completeness.reranking_mode == "answer_based":
            reranked_docs = self.remove_low_sim_docs(reranked_docs)

        system_prompt = get_prompt(
            cfg.kb_completeness.system_prompt, return_str=True
        )
        user_prompt = get_prompt(cfg.kb_completeness.user_prompt, return_str=True)

        # Key is the answer excerpt, Value is another dictionary with the document excerpts and similarity score.
        global_accepted_mappings = dict()
        matching_scores = list()  # Keep track of scores
        no_improvement_for = 0
        # Start the evaluation process with remaining documents using the assigned LLM.
        for i, step in enumerate(
            list(range(0, len(reranked_docs), self.page_size))
            if any(isinstance(item, (int, float)) for item in reranked_docs)
            else reranked_docs
        ):
            if isinstance(step, list):
                doc_batch = step
                # logger.debug(f"doc_batch size: {np.shape(doc_batch)}")
            else:
                doc_batch = reranked_docs[step : step + self.page_size]

            mappings = self.llm_document_analysis(
                system_prompt, user_prompt, question, agent_answer, doc_batch
            )

            global_accepted_mappings = self.extract_mapping_details(
                mappings, global_accepted_mappings
            )

            # Check if we have matched the answer fully with current mappings. Continue, if not.
            accepted_answer_excerpts = {
                key
                for key in global_accepted_mappings.keys()
                if np.max(global_accepted_mappings[key]["score"]) > self.score_threshold
            }
            answer_match_score = self.jaccard_similarity(
                "".join(accepted_answer_excerpts), agent_answer
            )

            logger.info(f"Current completeness score: {answer_match_score}")

            if answer_match_score > self.jaccard_similarity_threshold:
                logger.info("KB contents are enough to answer question.")
                break

            matching_scores.append(answer_match_score)

            if (
                i + 1 > self.evaluate_at
            ):  # Check if score has improved in the last `self.evaluate_at` iterations
                score_improvement = (
                    (matching_scores[i] - matching_scores[i - 1]) / matching_scores[i]
                    if matching_scores[i] > 0
                    else 0
                )
                no_improvement_for = (
                    no_improvement_for + 1
                    if score_improvement <= self.improvement_by
                    else 0
                )  # else restart count if improvement was recorded
                if no_improvement_for > self.stop_if_no_improvement_for:
                    logger.info(
                        f"No improvement of score recorded for {self.stop_if_no_improvement_for} iterations after iteration {self.evaluate_at}."
                    )
                    break

        return global_accepted_mappings

    def create_doc_clusters(
        self, embeddings: List[List[float]], docs: List[Document]
    ) -> Dict[int, List[Document]]:
        """Perfoms clustering of documents using the 2-D similarity matrix supplied with the K-means algorithm

        Args:
            embeddings (List[List[float]]): 2-D matrix of document embeddings, retrieved from the vector store.
            docs (List[Document]): List of Document objects to append to returned clusters.

        Returns:
            Dict[int, List[Document]]: Lists of Document objects, split in clusters.
        """
        # Perform KMeans clustering
        kmeans = KMeans(n_clusters=self.cluster_size)
        kmeans.fit(embeddings)

        # Get cluster labels
        cluster_labels = kmeans.labels_

        # Create clusters with original indices using defaultdict
        clusters = defaultdict(list)

        # Add the original indices to the corresponding clusters
        for i, label in enumerate(cluster_labels):
            clusters[label].append(docs[i])

        return list(clusters.values())

    def doc_reranking(
        self,
        docs: List[Document],
        answer: str | list[str],
        use_crossencoder: bool = cfg.kb_completeness.use_crossencoder,
        reranking_mode: str = cfg.kb_completeness.reranking_mode,
    ) -> List[Document]:
        reranked_docs = copy(docs)
        answer = [answer] if not isinstance(answer, list) else answer

        if not use_crossencoder:
            response = get_documents_by_id(
                os_index=cfg.opensearch.index,
                ids=[x.id for x in docs],
                _source=["vector_field"],
            )

            if len(response["docs"]) == 0:
                raise Exception("Documents supplied cannot be found in KB")

            vector_fields = [x["_source"]["vector_field"] for x in response["docs"]]
            if reranking_mode == "document_based":
                # Get clusters with original indices for each document in cluster
                clusters = self.create_doc_clusters(vector_fields, docs)
                reranked_docs = sorted(
                    clusters,
                    key=lambda cluster: np.mean(
                        [doc.metadata["score"] for doc in cluster]
                    ),
                    reverse=True,
                )
                # logger.debug(f"Batch sizes: {[len(docs) for docs in reranked_docs]}")

                return reranked_docs
            elif reranking_mode == "answer_based":
                for sentence in answer:
                    reranked_docs = self.embedding_based_re_ranking(
                        query=sentence,
                        doc_list=reranked_docs,
                        doc_embeddings=vector_fields,
                    )
            else:
                logger.debug("Reranking method not supported.")
                return reranked_docs
        elif use_crossencoder and reranking_mode == "answer_based":
            for sentence in answer:
                reranked_docs = self.cross_encoder_re_ranking(
                    query=sentence,
                    doc_list=reranked_docs,
                    remove_summary=True,
                    use_in_document_search=True,
                )

        # Sort based on the best match recorded (or the only one if single answer is supplied)
        reranked_docs = sorted(
            reranked_docs,
            reverse=True,
            key=lambda doc: np.max(doc.metadata["ce_score"])
            if "ce_score" in doc.metadata
            else 0,
        )

        logger.info(
            f"Documents reranked based on agent answer."
        )

        return reranked_docs

    def jaccard_similarity(self, sentence1: str, sentence2: str) -> float:
        def tokenize(text: str) -> set[str]:
            # Remove unwanted punctuation (except for letters, numbers, hyphens and apostrophes)
            text = re.sub(r"[^\w\s\'-]", " ", text.lower())
            words = text.split()
            lemmatized_words = [
                self._lemmatizer.lemmatize(word)
                for word in words
                if word.lower() not in self._stop_words
            ]  # Lemmatize each word
            return set(lemmatized_words)

        words1 = tokenize(sentence1)
        words2 = tokenize(sentence2)

        # Compute the Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union != 0 else 0

    @staticmethod
    def _remove_summary_from_chunk(page_content: str) -> str:
        return page_content.split("Chunk Content:")[-1]

    def _in_document_search(
        self, query: str, doc: Document, remove_summary: bool = True
    ):
        paragraph_list = (
            doc.page_content.split("\n\n")
            if not remove_summary
            else self._remove_summary_from_chunk(doc.page_content).split("\n\n")
        )
        cross_inp = [[query, paragraph] for paragraph in paragraph_list]

        cross_scores = self.cross_encoder.predict(cross_inp)

        return np.mean(cross_scores)

    @staticmethod
    def _update_doc_rank(
        doc_list: List[Document], new_scores: list[float]
    ) -> List[Document]:
        for rank, doc in enumerate(doc_list):
            cur_score = new_scores[rank]
            if "ce_score" in doc.metadata:
                doc.metadata["ce_score"] += [cur_score]
            else:
                doc.metadata["ce_score"] = [cur_score]

        return doc_list

    def embedding_based_re_ranking(
        self, query: str, doc_list: List[Document], doc_embeddings: list[float]
    ) -> List[Document]:
        embedding = self._get_embedding(query)

        cross_scores = cosine_similarity([embedding], doc_embeddings)[
            0
        ]  # 2-D array with a score for each document's similarity with query

        return self._update_doc_rank(doc_list, cross_scores)

    def cross_encoder_re_ranking(
        self,
        query: str,
        doc_list: List[Document],
        remove_summary: bool = True,
        use_in_document_search: bool = True,
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
        if not use_in_document_search:
            cross_inp = [
                [
                    query,
                    self._remove_summary_from_chunk(doc.page_content)
                    if remove_summary
                    else doc.page_content,
                ]
                for doc in doc_list
            ]
            cross_scores = self.cross_encoder.predict(cross_inp)
        else:
            cross_scores = [
                self._in_document_search(
                    query=query, doc=doc, remove_summary=remove_summary
                )
                for doc in doc_list
            ]

        return self._update_doc_rank(doc_list, cross_scores)


def kb_eval(
    qa_filepath: str,
    save_to_filepath: str,
    question_key: str = "input",
    answer_key: str = "expected_output",
    id_key: str = "ticket_id",
    items_key: str = "items",
    correct_prompt: bool = False,
):
    """Evalutes how complete is our KB given sets of Q&A pairs stored in file.

    Args:
        qa_filepath (str): Path to question and answer dataset in JSON.
        save_to_filepath (str): Where to save results from mappings.
        question_key (str, optional): Name of question key. Defaults to 'input'.
        answer_key (str, optional): Name of answer key. Defaults to 'expected_output'.
        id_key (str, optional): Identifier key to link output with input. Defaults to 'ticket_id'.
        items_key (str, optional): Where Q&A pairs are stored from the root object. Defaults to 'items'.
    """
    try:
        with open(qa_filepath) as f:
            dataset = json.load(f)
    except (json.JSONDecodeError, TypeError, ValueError, FileNotFoundError):
        raise typer.BadParameter(f"Could not read JSON file from {qa_filepath}")

    # Go through each Q&A pair and evaluate whether KB is complete enough to replicate A from Q.
    if items_key not in dataset:
        raise typer.BadParameter(f"Q&A pairs not found in {items_key}")

    mappings = dict()
    base_delay = 1  # Initial delay in seconds
    attempts = 0

    for pair in dataset[items_key]:
        kb_evaluator = KBEvaluator()  # Reset evaluator object for each Q&A pair
        logger.info(f"""Analyzing the pair with id: {pair[id_key]}""")

        question = pair[question_key]

        if cfg.prompt_preproc.correct_prompt:
            question = invoke_correcter(question)

        try:
            mappings[pair[id_key]] = kb_evaluator.evaluate_w_question(
                question=question, agent_answer=pair[answer_key]
            )
        except botocore.exceptions.ClientError:
            attempts += 1
            delay = base_delay * (2 ** (attempts - 1))
            logger.debug(
                f"ThrottlingException encountered. Retrying in {delay} seconds."
            )
            time.sleep(delay)
        except Exception as ex:
            logger.debug(traceback.format_exc())
            logger.error(
                f"Failed for pair with ticket id: {pair[id_key]} \n Exception: {ex}"
            )
            pass  # Skip for any other error.

        # Save to filepath - Overwrites in every loop execution
        save_fp = f"{save_to_filepath}{'.json' if not save_to_filepath.endswith('.json') else ''}"
        with open(save_fp, "w") as f:
            json.dump(mappings, f, indent=4)
