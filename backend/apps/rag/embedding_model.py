from typing import (
    Optional,
)

from apps.ollama.main import get_ollama_embedding_model_name_and_base_url
from apps.rag.utils import get_model_path
from config import (
    DEVICE_TYPE,
    RAG_EMBEDDING_MODEL_AUTO_UPDATE,
    RAG_EMBEDDING_MODEL_TRUST_REMOTE_CODE,
)
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Extra, Field, root_validator
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from ollama import AsyncClient, Client


def set_embedding_function(
    embedding_engine: Optional[str] = None,
    model_name: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    batch_size: Optional[int] = None,
    model_kwargs: dict = {
        "device": DEVICE_TYPE,
        "trust_remote_code": RAG_EMBEDDING_MODEL_TRUST_REMOTE_CODE,
    },
    update_model: bool = RAG_EMBEDDING_MODEL_AUTO_UPDATE,
) -> Embeddings:
    """
    Sets embedding function with LangChain Embeddings Interface based on
    the provider (embedding_engine) with the embedding model name and other
    settings attributes.

    Args:
        embedding_engine (Optional[str]): The provider used to set the embedding function.
        model_name (Optional[str]): The embedding model name.
        base_url (Optional[str]): The base url of the server that host the
            embedding models.
        api_key (Optional[str]): The api key to call the server.
        batch_size (Optional[int]): The batch size corresponds to the number
            of texts that is handled in one batch. Only used with "openai"
            embedding engine.
        model_kwargs (dict): The settings for the embedding model. Only used when
            no embedding engine provided (local execution with SentenceTransformers).

    Returns:
        Embeddings: Returns the LangChain Embedding Interface corresponding to the
            embedding engine.
    """
    match embedding_engine:
        case "ollama":
            model_name, base_url = get_ollama_embedding_model_name_and_base_url(
                model_name
            )
            return OllamaEmbeddings(model=model_name, base_url=base_url)
        case "openai":
            return OpenAIEmbeddings(
                model=model_name,
                base_url=base_url,
                api_key=api_key,
                chunk_size=batch_size,
            )
        case _:
            model_path = get_model_path(model=model_name, update_model=update_model)
            return HuggingFaceEmbeddings(
                model_name=model_path, model_kwargs=model_kwargs
            )


# TODO: To remove when new version of langchain-ollama is released (should be with v0.1.2)
class OllamaEmbeddings(BaseModel, Embeddings):
    """OllamaEmbeddings embedding model.

    Example:
        .. code-block:: python

            from langchain_ollama import OllamaEmbeddings

            embedder = OllamaEmbeddings(model="llama3")
            embedder.embed_query("what is the place that jonathan worked at?")
    """

    model: str
    """Model name to use."""

    base_url: Optional[str] = None
    """Base url the model is hosted under."""

    client_kwargs: Optional[dict] = {}
    """Additional kwargs to pass to the httpx Client. 
    For a full list of the params, see [this link](https://pydoc.dev/httpx/latest/httpx.Client.html)
    """

    _client: Client = Field(default=None)
    """
    The client to use for making requests.
    """

    _async_client: AsyncClient = Field(default=None)
    """
    The async client to use for making requests.
    """

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator(pre=False, skip_on_failure=True)
    def _set_clients(cls, values: dict) -> dict:
        """Set clients to use for ollama."""
        values["_client"] = Client(host=values["base_url"], **values["client_kwargs"])
        values["_async_client"] = AsyncClient(
            host=values["base_url"], **values["client_kwargs"]
        )
        return values

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed search docs."""
        embedded_docs = self._client.embed(self.model, texts)["embeddings"]
        return embedded_docs

    def embed_query(self, text: str) -> list[float]:
        """Embed query text."""
        return self.embed_documents([text])[0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed search docs."""
        embedded_docs = (await self._async_client.embed(self.model, texts))[
            "embeddings"
        ]
        return embedded_docs

    async def aembed_query(self, text: str) -> list[float]:
        """Embed query text."""
        return (await self.aembed_documents([text]))[0]
