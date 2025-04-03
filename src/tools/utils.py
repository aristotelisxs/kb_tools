import boto3
from botocore.config import Config
from langchain_core.prompts import ChatPromptTemplate
from langchain_aws import BedrockEmbeddings
from langchain_community.vectorstores import OpenSearchVectorSearch

from opensearchpy import OpenSearch
from tools.config import PROMPTS_DIR, cfg
from tools.log import logger


class Error(Exception):
    """Base class for exceptions in package"""

    pass


class PromptError(Error):
    """Prompt related errors"""

    pass


def combine_docs(docs) -> str:
    """Combine multiple documents into a single string"""

    return "\n".join(
        [
            "<documents>",
            *docs,
            "</documents>",
        ]
    )


def get_bedrock_client(is_eval=False):
    """Returns boto3 client for Bedrock API."""

    config = Config(
        connect_timeout=cfg.aws.connect_timeout,
        max_pool_connections=cfg.aws.max_pool_connections,
        read_timeout=cfg.aws.read_timeout,
        retries={
            "max_attempts": cfg.aws.retry_max_attempts,
            "mode": cfg.aws.retries_mode,
        },
    )

    client = boto3.client(
        "bedrock-runtime",
        aws_access_key_id=cfg.aws.access_key_id,
        aws_secret_access_key=cfg.aws.secret_access_key.get_secret_value(),
        region_name=cfg.aws.region if not is_eval else cfg.evaluation.region,
        config=config,
    )

    return client


def get_opensearch_client(index: str | None = cfg.opensearch.index):
    return OpenSearch(
        hosts=[cfg.opensearch.url],
        http_auth=(cfg.opensearch.user, cfg.opensearch.password.get_secret_value()),
        use_ssl=False,
        verify_certs=False,
        http_compress=True,
        index=index,
    )


def get_vector_store(index: str | None = None):
    if index is None:
        index = cfg.opensearch.index

    vector_store = OpenSearchVectorSearch(
        opensearch_url=cfg.opensearch.url,
        index_name=index,
        embedding_function=get_embedding(),
        http_auth=(cfg.opensearch.user, cfg.opensearch.password.get_secret_value()),
        use_ssl=False,
        verify_certs=False,
        ssl_assert_hostname=False,
        ssl_show_warn=False,
    )

    return vector_store


def get_prompt(prompt_name: str, return_str: bool = False) -> ChatPromptTemplate | str:
    """Return ChatPromptTemplate from disk"""

    prompt = get_prompt_from_disk(prompt_name)

    if prompt is None:
        raise PromptError(f"Could not find prompt {prompt_name}")

    if return_str:
        return prompt

    return ChatPromptTemplate.from_template(prompt)


def get_prompt_from_disk(prompt_name: str) -> str | None:
    """Return prompt from disk"""

    prompt_path = PROMPTS_DIR.joinpath(f"{prompt_name}.md")
    if not prompt_path.exists():
        logger.warning("Prompt file does not exist: %s", prompt_path)
        return None

    return prompt_path.read_text()


def get_embedding():
    embedding = BedrockEmbeddings(
        client=get_bedrock_client(), model_id=cfg.bedrock.embedding_model
    )

    return embedding
