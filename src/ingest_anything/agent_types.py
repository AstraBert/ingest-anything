from llama_index.core.llms.llm import LLM
from llama_index.core.agent.workflow import FunctionAgent, ReActAgent
from llama_index.core.tools import BaseTool, QueryEngineTool
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.indices.query.query_transform.base import (
    StepDecomposeQueryTransform,
)
from llama_index.core.query_engine import TransformQueryEngine, MultiStepQueryEngine
from typing import Callable, Awaitable, List, Optional, Literal

try:
    from .ingestion import (
        IngestAnything,
        IngestCode,
        BasePydanticVectorStore,
        BaseReader,
        VectorStoreIndex,
    )
except ImportError:
    from ingestion import (
        IngestAnything,
        IngestCode,
        BasePydanticVectorStore,
        BaseReader,
        VectorStoreIndex,
    )


class IngestAnythingFunctionAgent(IngestAnything):
    """
    IngestAnythingFunctionAgent is a specialized agent class for ingesting data into a vector database and providing advanced query capabilities using LLMs and optional query transformations.
    Args:
        vector_database (BasePydanticVectorStore): The vector database to use for storing and querying embeddings.
        llm (LLM): The large language model used for query processing and transformations.
        reader (Optional[BaseReader], optional): Optional reader for ingesting data. Defaults to None.
        tools (Optional[List[BaseTool | Callable | Awaitable]], optional): Additional tools to be made available to the agent. Defaults to None.
        query_transform (Optional[Literal["hyde", "multi_step"]], optional): Optional query transformation strategy. Can be "hyde" for HyDEQueryTransform or "multi_step" for StepDecomposeQueryTransform. Defaults to None.
    Attributes:
        llm (LLM): The language model used for query processing.
        query_transform (Optional[str]): The query transformation strategy.
        tools (List[BaseTool | Callable | Awaitable]): List of tools available to the agent.
        vector_store_index: The index of the ingested data in the vector store.
        query_engine: The query engine instance.
        query_engine_tool: The tool for querying the vector database.
    """

    def __init__(
        self,
        vector_database: BasePydanticVectorStore,
        llm: LLM,
        reader: Optional[BaseReader] = None,
        tools: Optional[List[BaseTool | Callable | Awaitable]] = None,
        query_transform: Optional[Literal["hyde", "multi_step"]] = None,
    ) -> None:
        super().__init__(vector_store=vector_database, reader=reader)
        self.llm = llm
        self.query_transform = query_transform
        if tools is None:
            tools = []
        self.tools = tools
        try:
            self.vector_store.get_nodes()
        except Exception:
            self.vector_store_index = None
        else:
            self.vector_store_index = VectorStoreIndex.from_vector_store(self.vector_store)

    def ingest(
        self,
        files_or_dir: str | List[str],
        embedding_model: str,
        chunker: Literal[
            "token", "sentence", "semantic", "sdpm", "late", "neural", "slumber"
        ],
        tokenizer: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        min_characters_per_chunk: Optional[int] = None,
        min_sentences: Optional[int] = None,
        slumber_genie: Optional[Literal["openai", "gemini"]] = None,
        slumber_model: Optional[str] = None,
    ):
        """
        Ingests files or directories into a vector store index using the specified embedding model and chunking strategy.

        Args:
            files_or_dir (str | List[str]): Path to a file, directory, or list of files/directories to ingest.
            embedding_model (str): Name of the embedding model to use for vectorization.
            chunker (Literal["token", "sentence", "semantic", "sdpm", "late", "neural", "slumber"]): Chunking strategy to use for splitting the text.
            tokenizer (Optional[str], optional): Name of the tokenizer to use. Defaults to None.
            chunk_size (Optional[int], optional): Size of each chunk. Defaults to None.
            chunk_overlap (Optional[int], optional): Number of overlapping tokens or sentences between chunks. Defaults to None.
            similarity_threshold (Optional[float], optional): Minimum similarity threshold for semantic chunking. Defaults to None.
            min_characters_per_chunk (Optional[int], optional): Minimum number of characters per chunk. Defaults to None.
            min_sentences (Optional[int], optional): Minimum number of sentences per chunk. Defaults to None.
            slumber_genie (Optional[Literal["openai", "gemini"]]):
            The LLM provider for the SlumberChunker. Defaults to "openai".slumber_model (Optional[str]):
            The Gemini model name to use for "slumber" chunking. Defaults to "gemini-2.0-flash" or "gpt-4.1" (based on the "slumber_genie" choice) if not specified and "slumber" is chosen.

        Returns:
            None
        """
        self.vector_store_index = super().ingest(
            files_or_dir,
            embedding_model,
            chunker,
            tokenizer,
            chunk_size,
            chunk_overlap,
            similarity_threshold,
            min_characters_per_chunk,
            min_sentences,
            slumber_genie,
            slumber_model,
        )

    def _get_query_engine_tool(self) -> None:
        query_engine = self.vector_store_index.as_query_engine(llm=self.llm)
        if self.query_transform == "hyde":
            qt = HyDEQueryTransform(llm=self.llm)
            self.query_engine = TransformQueryEngine(
                query_engine=query_engine, query_transform=qt
            )
        elif self.query_transform == "multi_step":
            qt = StepDecomposeQueryTransform(llm=self.llm, verbose=True)
            self.query_engine = MultiStepQueryEngine(
                query_engine=query_engine, query_transform=qt
            )
        else:
            self.query_engine = query_engine
        self.query_engine_tool = QueryEngineTool.from_defaults(
            query_engine=self.query_engine,
            name="query_engine_tool",
            description="Retrieves information from a vector database",
        )

    def get_agent(
        self,
        name: str = "FunctionAgent",
        description: str = "A useful AI agent",
        system_prompt: str = "You are a useful assistant who uses the tools available to you whenever it is needed",
    ) -> FunctionAgent:
        """
        Creates and returns a FunctionAgent instance with the specified name, description, and system prompt.

        This method prepares the agent by combining existing tools with a query engine tool, then initializes
        a FunctionAgent using these tools and the provided parameters.

        Args:
            name (str): The name of the agent. Defaults to "FunctionAgent".
            description (str): A brief description of the agent. Defaults to "A useful AI agent".
            system_prompt (str): The system prompt to guide the agent's behavior. Defaults to a helpful assistant prompt.

        Returns:
            FunctionAgent: An instance of FunctionAgent configured with the specified parameters and tools.
        """
        self._get_query_engine_tool()
        agent_tools = self.tools + [self.query_engine_tool]
        agent = FunctionAgent(
            name=name,
            description=description,
            system_prompt=system_prompt,
            tools=agent_tools,
        )
        return agent


class IngestCodeFunctionAgent(IngestCode):
    """A class that combines code ingestion with function agent capabilities.
    This class extends IngestCode to provide functionality for creating agents that can query
    and interact with ingested code using various tools and query transformation methods.
    Parameters
    ----------
    vector_database : BasePydanticVectorStore
        The vector database to store and retrieve embeddings.
    llm : LLM
        The language model to use for queries and transformations.
    tools : Optional[List[BaseTool | Callable | Awaitable]], default=None
        Additional tools to be used by the agent.
    query_transform : Optional[Literal["hyde", "multi_step"]], default=None
        The type of query transformation to apply. Options are "hyde" or "multi_step".
    Methods
    -------
    ingest(files, embedding_model, language, return_type, tokenizer, chunk_size, include_nodes)
        Ingests code files into the vector database.
    _get_query_engine_tool()
        Initializes the query engine with specified transformations.
    get_agent(name, description, system_prompt)
        Creates and returns a FunctionAgent with configured tools.
    """

    def __init__(
        self,
        vector_database: BasePydanticVectorStore,
        llm: LLM,
        tools: Optional[List[BaseTool | Callable | Awaitable]] = None,
        query_transform: Optional[Literal["hyde", "multi_step"]] = None,
    ) -> None:
        """Initialize the IngestCodeFunctionAgent.
        Parameters
        ----------
        vector_database : BasePydanticVectorStore
            The vector database to store and retrieve embeddings.
        llm : LLM
            The language model to use for queries and transformations.
        tools : Optional[List[BaseTool | Callable | Awaitable]], default=None
            Additional tools to be used by the agent.
        query_transform : Optional[Literal["hyde", "multi_step"]], default=None
            The type of query transformation to apply.
        """
        super().__init__(vector_store=vector_database)
        self.llm = llm
        self.query_transform = query_transform
        if tools is None:
            tools = []
        self.tools = tools
        try:
            self.vector_store.get_nodes()
        except Exception:
            self.vector_store_index = None
        else:
            self.vector_store_index = VectorStoreIndex.from_vector_store(self.vector_store)


    def ingest(
        self,
        files: List[str],
        embedding_model: str,
        language: str,
        return_type: Optional[Literal["chunks", "texts"]] = None,
        tokenizer: Optional[str] = None,
        chunk_size: Optional[int] = None,
        include_nodes: Optional[bool] = None,
    ):
        """Ingest code files into the vector database.
        Parameters
        ----------
        files : List[str]
            List of file paths to ingest.
        embedding_model : str
            Name or path of the embedding model to use.
        language : str
            Programming language of the code files.
        return_type : Optional[Literal["chunks", "texts"]], default=None
            Type of return value from ingestion.
        tokenizer : Optional[str], default=None
            Tokenizer to use for text splitting.
        chunk_size : Optional[int], default=None
            Size of text chunks for splitting.
        include_nodes : Optional[bool], default=None
            Whether to include node information.
        Returns
        -------
        VectorStoreIndex
            The index created from ingested files.
        """
        self.vector_store_index = super().ingest(
            files,
            embedding_model,
            language,
            return_type,
            tokenizer,
            chunk_size,
            include_nodes,
        )

    def _get_query_engine_tool(self) -> None:
        query_engine = self.vector_store_index.as_query_engine(llm=self.llm)
        if self.query_transform == "hyde":
            qt = HyDEQueryTransform(llm=self.llm)
            self.query_engine = TransformQueryEngine(
                query_engine=query_engine, query_transform=qt
            )
        elif self.query_transform == "multi_step":
            qt = StepDecomposeQueryTransform(llm=self.llm, verbose=True)
            self.query_engine = MultiStepQueryEngine(
                query_engine=query_engine, query_transform=qt
            )
        else:
            self.query_engine = query_engine
        self.query_engine_tool = QueryEngineTool.from_defaults(
            query_engine=self.query_engine,
            name="query_engine_tool",
            description="Retrieves information from a vector database containing code snippets",
        )

    def get_agent(
        self,
        name: str = "FunctionAgent",
        description: str = "A useful AI agent",
        system_prompt: str = "You are a useful assistant who uses the tools available to you whenever it is needed",
    ) -> FunctionAgent:
        """Create and return a FunctionAgent with configured tools.
        Parameters
        ----------
        name : str, default="FunctionAgent"
            Name of the agent.
        description : str, default="A useful AI agent"
            Description of the agent's purpose.
        system_prompt : str, default="You are a useful assistant who uses the tools available to you whenever it is needed"
            System prompt for the agent.
        Returns
        -------
        FunctionAgent
            Configured function agent with query engine and additional tools.
        """
        self._get_query_engine_tool()
        agent_tools = self.tools + [self.query_engine_tool]
        agent = FunctionAgent(
            llm=self.llm,
            name=name,
            description=description,
            system_prompt=system_prompt,
            tools=agent_tools,
        )
        return agent


class IngestAnythingReActAgent(IngestAnythingFunctionAgent):
    """A ReAct agent implementation for ingesting and processing data.

    This class extends IngestAnythingFunctionAgent to provide ReAct (Reasoning and Acting)
    agent capabilities for data ingestion tasks.
    """

    def get_agent(
        self,
        name="ReActAgent",
        description="A useful AI agent",
        system_prompt="You are a useful assistant who uses the tools available to you whenever it is needed",
    ) -> ReActAgent:
        """Creates and returns a configured ReAct agent instance.

        Args:
            name (str, optional): Name of the agent. Defaults to "ReActAgent".
            description (str, optional): Description of the agent. Defaults to "A useful AI agent".
            system_prompt (str, optional): System prompt for the agent. Defaults to basic assistant prompt.

        Returns:
            ReActAgent: Configured ReAct agent instance with query engine and tools.

        Note:
            This method automatically adds the query engine tool to the agent's available tools.
        """
        self._get_query_engine_tool()
        agent_tools = self.tools + [self.query_engine_tool]
        agent = ReActAgent(
            llm=self.llm,
            name=name,
            description=description,
            system_prompt=system_prompt,
            tools=agent_tools,
        )
        return agent


class IngestCodeReActAgent(IngestCodeFunctionAgent):
    """
    A class that inherits from IngestCodeFunctionAgent to create a ReAct agent for code ingestion.

    The ReAct agent combines reasoning and acting capabilities to handle code-related tasks.

    Attributes:
        Inherits all attributes from IngestCodeFunctionAgent
    """

    def get_agent(
        self,
        name="ReActAgent",
        description="A useful AI agent",
        system_prompt="You are a useful assistant who uses the tools available to you whenever it is needed",
    ) -> ReActAgent:
        """
        Creates and returns a configured ReActAgent instance with the specified parameters
            and available tools.

        Args:
            name (str, optional): The name of the agent. Defaults to "ReActAgent".
            description (str, optional): Description of the agent's purpose.
                Defaults to "A useful AI agent".
            system_prompt (str, optional): The system prompt for the agent.
                Defaults to "You are a useful assistant...".

        Returns:
            ReActAgent: A configured ReAct agent instance with query engine and other tools.
        """
        self._get_query_engine_tool()
        agent_tools = self.tools + [self.query_engine_tool]
        agent = ReActAgent(
            llm=self.llm,
            name=name,
            description=description,
            system_prompt=system_prompt,
            tools=agent_tools,
        )
        return agent
