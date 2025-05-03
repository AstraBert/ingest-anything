## ingest_anything package

### Submodules

#### ingest\_anything.add\_types module

This module defines the data models for configuring text chunking and ingestion inputs.

*   **Classes**

    *   `Chunking(BaseModel)`: A Pydantic model for configuring text chunking parameters.

        *   Inherits from: `pydantic.BaseModel`
        *   Description: This class defines the configuration for different text chunking strategies and their associated parameters.
        *   Attributes:
            *   `chunker` (`Literal["token", "sentence", "semantic", "sdpm", "late", "slumber", "neural"]`): The chunking strategy to use.
                *   `"token"`: Split by number of tokens
                *   `"sentence"`: Split by sentences
                *   `"semantic"`: Split by semantic similarity
                *   `"sdpm"`: Split using sentence distance probability matrix
                *   `"late"`: Delayed chunking strategy
                *   `"slumber"`: LLM-based chunking using Gemini
                *   `"neural"`: Finetuned-for-chunking BERT-based chunking
            *   `chunk_size` (`Optional[int]`): The target size for each chunk. Defaults to 512 if not specified.
            *   `chunk_overlap` (`Optional[int]`): The number of overlapping units between consecutive chunks. Defaults to 128 if not specified.
            *   `similarity_threshold` (`Optional[float]`): The minimum similarity threshold for semantic and SDPM chunking. Defaults to 0.7 if not specified.
            *   `min_characters_per_chunk` (`Optional[int]`): The minimum number of characters required for a valid chunk. Defaults to 24 if not specified.
            *   `min_sentences` (`Optional[int]`): The minimum number of sentences required for a valid chunk. Defaults to 1 if not specified.
            *   `gemini_model` (`Optional[str]`): The Gemini model name to use for "slumber" chunking. Defaults to "gemini-2.0-flash" if not specified and "slumber" is chosen.
        *   Example:

            ```python
            >>> from ingest_anything.add_types import Chunking
            >>> chunking_config = Chunking(chunker="semantic", chunk_size=256, chunk_overlap=64, similarity_threshold=0.8, min_characters_per_chunk=50, min_sentences=2)
            ```

    *   `CodeFiles(BaseModel)`: A Pydantic model for validating and processing lists of code file paths.

        *   Inherits from: `pydantic.BaseModel`
        *   Description: This class extends BaseModel to handle file path validation, ensuring that all provided paths exist in the filesystem.
        *   Attributes:
            *   `files` (`List[str]`): A list of file paths to be validated.
        *   Raises:
            *   `ValueError`: When none of the provided file paths exist in the filesystem.
        *   Example:

            ```python
            >>> from ingest_anything.add_types import CodeFiles
            >>> code_files = CodeFiles(files=["file1.py", "file2.py"])
            ```

    *   `CodeChunking(BaseModel)`: A Pydantic model for configuring code chunking parameters.

        *   Inherits from: `pydantic.BaseModel`
        *   Description: This class handles the configuration and validation of parameters used for chunking code into smaller segments, with support for different programming languages and tokenization methods.
        *   Attributes:
            *   `language` (`str`): The programming language of the code to be chunked.
            *   `return_type` (`Optional[Literal["chunks", "texts"]]`): The format of the chunked output. Defaults to "chunks" if not specified.
            *   `tokenizer` (`Optional[str]`): The name of the tokenizer to use. Defaults to "gpt2".
            *   `chunk_size` (`Optional[int]`): The maximum size of each chunk in tokens. Defaults to 512.
            *   `include_nodes` (`Optional[bool]`): Whether to include AST nodes in the output. Defaults to False.
        *   Example:

            ```python
            >>> from ingest_anything.add_types import CodeChunking
            >>> code_chunking_config = CodeChunking(language="python", return_type="chunks", tokenizer="gpt2", chunk_size=256, include_nodes=True)
            ```

    *   `IngestionInput(BaseModel)`: A class that validates and processes ingestion inputs for document processing.

        *   Inherits from: `pydantic.BaseModel`
        *   Description: This class handles different types of document inputs and chunking strategies, converting files and setting up appropriate chunking mechanisms based on the specified configuration.
        *   Attributes:
            *   `files_or_dir` (`Union[str, List[str]]`): Path to directory containing files or list of file paths to process
            *   `chunking` (`Chunking`): Configuration for the chunking strategy to be used
            *   `tokenizer` (`Optional[str]`, default=`None`): Name or path of the tokenizer model to be used (required for 'token' and 'sentence' chunking)
            *   `embedding_model` (`str`): Name or path of the embedding model to be used
        *   Example:

            ```python
            >>> from ingest_anything.add_types import IngestionInput, Chunking
            >>> ingestion_config = IngestionInput(
            ...     files_or_dir="path/to/documents",
            ...     chunking=Chunking(chunker="token", chunk_size=256, chunk_overlap=64),
            ...     tokenizer="bert-base-uncased",
            ...     embedding_model="sentence-transformers/all-mpnet-base-v2"
            ... )
            ```

#### ingest\_anything.ingestion module

This module defines the `IngestAnything` and `IngestCode` classes, which handle the ingestion and storage of documents and code files in a Qdrant vector database.

*   **Classes**

    *   `IngestAnything`: Provides a high-level interface for ingesting documents, chunking them using various strategies, and indexing them into a vector store for semantic search.

        *   `__init__(vector_store: BasePydanticVectorStore, reader: Optional[BaseReader] = None)`
            *   Parameters:
                *   `vector_store` (`BasePydanticVectorStore`): The vector store instance where document embeddings will be stored.
                *   `reader` (`Optional[BaseReader]`, default=`None`): Optional custom document reader. If not provided, a default DoclingReader is used.
            *   Example:

                ```python
                >>> from llama_index.vector_stores.qdrant import QdrantVectorStore
                >>> from ingest_anything.ingestion import IngestAnything
                >>> vector_store = QdrantVectorStore(client=client, collection_name="my_collection")
                >>> ingestor = IngestAnything(vector_store=vector_store)
                ```

        *   `ingest(files_or_dir: str | List[str], embedding_model: str, chunker: Literal["token", "sentence", "semantic", "sdpm", "late", "neural", "slumber"], tokenizer: Optional[str] = None, chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None, similarity_threshold: Optional[float] = None, min_characters_per_chunk: Optional[int] = None, min_sentences: Optional[int] = None, gemini_model: Optional[str] = None) -> VectorStoreIndex`
            *   Description: Ingest documents from files or directories using the specified chunking strategy and create a searchable vector index.
            *   Parameters:
                *   `files_or_dir` (`str` or `List[str]`): Path to file(s) or directory to ingest.
                *   `embedding_model` (`str`): Name of the embedding model to use: supports OpenAI, HuggingFace, Cohere, Jina AI and Model2Vec
                *   `chunker` (`Literal["token", "sentence", "semantic", "sdpm", "late", "neural", "slumber"]`): Chunking strategy to use.
                *   `tokenizer` (`str`, optional): Tokenizer to use for chunking.
                *   `chunk_size` (`int`, optional): Size of chunks.
                *   `chunk_overlap` (`int`, optional): Number of overlapping tokens/sentences between chunks.
                *   `similarity_threshold` (`float`, optional): Similarity threshold for semantic chunking.
                *   `min_characters_per_chunk` (`int`, optional): Minimum number of characters per chunk.
                *   `min_sentences` (`int`, optional): Minimum number of sentences per chunk.
                *   `gemini_model` (`str`, optional): Name of Gemini model to use for chunking, if applicable.
            *   Returns:
                *   `VectorStoreIndex`: Index containing the ingested and embedded document chunks.
            *   Example:

                ```python
                >>> from llama_index.vector_stores.qdrant import QdrantVectorStore
                >>> from ingest_anything.ingestion import IngestAnything
                >>> vector_store = QdrantVectorStore(client=client, collection_name="my_collection")
                >>> ingestor = IngestAnything(vector_store=vector_store)
                >>> index = ingestor.ingest(
                ...     files_or_dir="path/to/documents",
                ...     embedding_model="sentence-transformers/all-mpnet-base-v2",
                ...     chunker="semantic",
                ...     similarity_threshold=0.8
                ... )
                ```

    *   `IngestCode`:  is a class for ingesting code files, chunking them, embedding the chunks, and storing them in a vector store for efficient search and retrieval.

        *   `__init__(vector_store: BasePydanticVectorStore)`
            *   Parameters:
                *   `vector_store` (`BasePydanticVectorStore`): The vector store instance where embedded code chunks will be stored.
            *   Example:
                ```python
                >>> from llama_index.vector_stores.qdrant import QdrantVectorStore
                >>> from ingest_anything.ingestion import IngestCode
                >>> vector_store = QdrantVectorStore(client=client, collection_name="my_code_collection")
                >>> ingestor = IngestCode(vector_store=vector_store)
                ```

        *   `ingest(files: List[str], embedding_model: str, language: str, return_type: Optional[Literal["chunks", "texts"]] = None, tokenizer: Optional[str] = None, chunk_size: Optional[int] = None, include_nodes: Optional[bool] = None)`
            *   Description: Ingest code files and create a searchable vector index.
            *   Parameters:
                *   `files` (`List[str]`): List of file paths to ingest
                *   `embedding_model` (`str`): Name of the HuggingFace embedding model to use
                *   `language` (`str`): Programming language of the code files
                *   `return_type` (`Literal["chunks", "texts"]`, optional): Type of return value from chunking
                *   `tokenizer` (`str`, optional): Name of tokenizer to use
                *   `chunk_size` (`int`, optional): Size of chunks for text splitting
                *   `include_nodes` (`bool`, optional): Whether to include AST nodes in chunking
            *   Returns:
                *   `VectorStoreIndex`: Index containing the ingested and embedded code chunks
            *   Example:
                ```python
                >>> from llama_index.vector_stores.qdrant import QdrantVectorStore
                >>> from ingest_anything.ingestion import IngestCode
                >>> vector_store = QdrantVectorStore(client=client, collection_name="my_code_collection")
                >>> ingestor = IngestCode(vector_store=vector_store)
                >>> index = ingestor.ingest(
                ...     files=["file1.py", "file2.py"],
                ...     embedding_model="sentence-transformers/all-mpnet-base-v2",
                ...     language="python",
                ...     chunk_size=256
                ... )
                ```