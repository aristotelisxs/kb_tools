import os

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).absolute().parent
ROOT_DIR = BASE_DIR.parents[1]
INFRA_DIR = ROOT_DIR.joinpath("infra")
PROMPTS_DIR = BASE_DIR.joinpath("prompts")

DOTENV_PATH = INFRA_DIR.joinpath("tools.env")


class OpenSearchConfig(BaseModel):
    url: str = Field(default="http://localhost:9200")
    user: str
    password: SecretStr
    index: str


class BedrockModel(BaseModel):
    bedrock_model_id: str = Field()
    temperature: float | None = Field(default=None, ge=0.0, le=1.0)
    max_tokens: int | None = Field(default=None, gt=0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=0, le=500)
    stop_sequences: list[str] | None = Field(default=None)

    def get_model_kwargs(self) -> dict[str, Any]:
        """Get model-specific kwargs with appropriate defaults."""
        base_params = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "stop_sequences": self.stop_sequences or [],
        }

        if "anthropic" in self.bedrock_model_id.lower():
            # see https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages.html
            kwargs = {
                "temperature": base_params["temperature"] or 1.0,
                # 1024 is default on langchain_aws. we reproduce this default behaviour
                "max_tokens": base_params["max_tokens"] or 1024,
                "top_p": base_params["top_p"] or 1,
                "stop_sequences": base_params["stop_sequences"],
            }

            if self.top_k:
                kwargs["top_k"] = self.top_k

            return kwargs
        else:
            raise ValueError(f"Unsupported model id: {self.bedrock_model_id}")


class BedrockModelIds:
    SONNET: str = "anthropic.claude-3-5-sonnet-20240620-v1:0"
    HAIKU: str = "anthropic.claude-3-haiku-20240307-v1:0"
    

class CorrectingConfig(BedrockModel):
    bedrock_model_id: str = Field(default=BedrockModelIds.HAIKU)
    temperature: float = Field(default=0.1)
    max_tokens: int = Field(default=1024)


class AWSConfig(BaseModel):
    access_key_id: str
    secret_access_key: SecretStr
    region: str = Field(default="eu-central-1")
    connect_timeout: int = Field(default=20)
    max_pool_connections: int = Field(default=200)
    read_timeout: int = Field(default=180)
    retries_mode: str = Field(default="adaptive")
    retry_max_attempts: int = Field(default=3)


class RetrieverConfig(BaseModel):
    min_docs: int = Field(
        default=5, description="Min docs to return after threshold filtering"
    )


class CommonPromptTagsConfig(BaseModel):
    sketchpad: str = Field(default="sketchpad")
    thinking: str = Field(default="thinking")
    chain_of_thought: str = Field(default="chain_of_thought")
    instructions: str = Field(default="instructions")

    system: str = Field(default="system")
    human: str = Field(default="human")


class KBCompletenessConfig(BaseModel):
    system_prompt: str = Field(default="kb_completeness")
    user_prompt: str = Field(default="kb_completeness_human")

    # Model-specific
    model_id: str = Field(default=BedrockModelIds.SONNET)
    query_embed_model: str = Field(default=BedrockModelIds.TITAN_EMBED)
    temperature: float = Field(default=0.0)
    max_tokens: int = Field(default=2048)

    reranking_mode: str = Field(
        default="document_based"
    )  # allowed values: ['document_based', 'answer_based']
    use_crossencoder: bool = Field(default=False)
    # No effect if use_crossencoder is False
    use_in_document_search: bool = Field(default=True)

    # Process-specific
    docs_to_retrieve: int = Field(default=100)
    docs_in_prompt: int = Field(default=15)
    cluster_size: int = Field(default=10)

    excerpt_score_threshold: float = Field(default=0.5)
    use_reranking: bool = Field(default=True)
    evaluate_at: int = Field(default=2)
    improvement_by: float = Field(default=0.05)
    stop_if_no_improvement_for: int = Field(default=2)
    completeness_score_threshold: float = Field(default=0.8)


class PromptPreprocConfig(BaseModel):
    correct_prompt: bool = False
    correcting_prompt_name: str = Field(default="correcting")
    do_spell_checking: bool = Field(default=False)
    check_jaccard_similarity: bool = Field(default=True)


class Config(BaseSettings):
    aws: AWSConfig
    opensearch: OpenSearchConfig
    retriever: RetrieverConfig = RetrieverConfig()
    kb_completeness: KBCompletenessConfig = KBCompletenessConfig()
    common_prompt_tags: CommonPromptTagsConfig = CommonPromptTagsConfig()
    correcting_model: CorrectingConfig = CorrectingConfig()
    prompt_preproc: PromptPreprocConfig = PromptPreprocConfig()

    model_config = SettingsConfigDict(
        env_file=DOTENV_PATH,
        env_prefix="CB_",
        env_nested_delimiter="__",
        nested_model_default_partial_update=True,
        extra="ignore",
    )


cfg = Config()
