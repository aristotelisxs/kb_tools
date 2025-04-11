import re
from functools import partial
from operator import is_not
from typing import Any, Coroutine

import nltk
from langchain_aws import ChatBedrock
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.runnables import RunnableConfig
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker

from tools.config import cfg
from tools.log import logger


def get_llm_input_text(input: LanguageModelInput) -> str:
    text = input
    if isinstance(input, str):
        text = input
    elif isinstance(input, ChatPromptValue):
        if len(input.messages) > 2:
            raise ValueError(f"Too much messages for {input} input")
        if not isinstance(input.messages[1], HumanMessage):
            raise ValueError(f"{input.messages[1]} is not a human message")
        text = input.messages[1].content
    else:
        raise ValueError(f"Unsupported type of input {input}")

    return text


class Correcter(ChatBedrock):
    """LLM input (by end-user) grammatical correcter. Does not change the meaning or structure of the input,
    and makes exact keyword match more accurate by using corrected terms."""

    jaccard_similarity_threshold: float = 0.5
    # Super conservative to avoid input corruption - Only checks if a single character could be replaced to match known words
    spell: SpellChecker = SpellChecker(distance=1)

    def model_post_init(self, __context: Any) -> None:
        nltk.download("wordnet")
        nltk.download("stopwords")
        self._lemmatizer = WordNetLemmatizer()
        self._stop_words = set(stopwords.words("english"))

    def invoke(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> BaseMessage:
        if cfg.prompt_preproc.do_spell_checking:
            input = self.spell_check(input)

        reply = super().invoke(
            input,
            config=config,
            **kwargs,
        )

        return self.correct_query(input, reply)

    async def ainvoke(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Coroutine[Any, Any, BaseMessage]:
        if cfg.prompt_preproc.do_spell_checking:
            input = self.spell_check(input)

        reply = await super().ainvoke(
            input,
            config=config,
            **kwargs,
        )

        return self.correct_query(input, reply)

    def check_word(self, word: str) -> str:
        # Check if the word exists in the dictionary
        known_word = " ".join(self.spell.known([word]))

        if known_word != "":
            return known_word.lower()
        # If the word is not known but it is capitalized, it's probably a game-related term
        elif known_word == "" and not word.isupper():
            # If not, find the closest word
            nearest_word = self.spell.correction(word)
            if nearest_word is None:
                return word.lower()
            return nearest_word.lower()
        # if all else fails
        return word.lower()

    def spell_check(self, input: LanguageModelInput) -> LanguageModelInput:
        text = get_llm_input_text(input)

        text = re.sub(r"[^\w\s\'-]", " ", text)
        words = text.split()
        words = [self.check_word(word) for word in words if word is not None]
        words = list(filter(partial(is_not, None), words))

        text = " ".join(words)

        if isinstance(input, ChatPromptValue):
            input.messages[1].content = text

        return input

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

    def correct_query(
        self,
        query: LanguageModelInput,
        reply: BaseMessage,
    ) -> BaseMessage | str:
        content = str(reply.content).strip()

        corrected = ""
        # Look for text as assumed by the prompt
        text_match = re.search(r"<text>(.*?)</text>", content, re.DOTALL)
        if text_match:
            corrected = text_match.group(1).strip()

        query = get_llm_input_text(query)

        # Handle empty or None cases. Nothing should be corrected
        if not corrected or corrected.isspace():
            logger.debug("Corrected text is empty. Returning original query")
            return query

        elif corrected and cfg.prompt_preproc.check_jaccard_similarity:
            sim_score = self.jaccard_similarity(query, corrected)
            if sim_score < self.jaccard_similarity_threshold:
                # Reject the corrected input
                logger.debug(
                    "Corrected text rejected by jaccard_similarity. Returning original query"
                )
                return query
            # If jaccard_similarity passes, then return corrected
            return corrected
        else:
            return corrected
