from typing import List, Optional, Dict
from datetime import datetime
from dataclasses import dataclass
import os
import tempfile
import shutil
import logging
import chromadb # type: ignore
import logging
from chromadb.config import Settings # type: ignore

# Updated imports to use newer packages
from langchain_community.document_loaders import TextLoader # type: ignore
from langchain_text_splitters import RecursiveCharacterTextSplitter # type: ignore
from langchain_chroma import Chroma # type: ignore
from langchain_huggingface import HuggingFaceEmbeddings # type: ignore
from langchain.docstore.document import Document # type: ignore

@dataclass
class DocumentMetadata:
    version: str
    date: datetime
    security_tag: str
    access_tags: List[str]

class EmbeddingFunctionWrapper:
    def __init__(self, embedding_function):
        self.embedding_function = embedding_function
    
    def __call__(self, input):
        return self.embedding_function.embed_documents(input)

class LangChainPDFProcessor:
    def __init__(self, persist_directory: str = "db"):
        """Initialize the document processor with LangChain components."""
        self.persist_directory = persist_directory
        
        # Clean up existing directory if it exists
        if os.path.exists(persist_directory):
            os.system(f"rm -rf {persist_directory}")
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize embeddings with explicit model name
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        self.embedding_function = EmbeddingFunctionWrapper(self.embeddings)
        
        # Initialize ChromaDB with updated settings
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                is_persistent=True
            )
        )
        
        # Initialize vector store
        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings,
            client=self.client,
            collection_metadata={"hnsw:space": "cosine"}
        )

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )

    def process_document(self, 
                        file_path: str, 
                        metadata: DocumentMetadata) -> List[Document]:
        """Process a text file and return LangChain documents with metadata."""
        try:
            # Load and process the text file
            loader = TextLoader(file_path)
            documents = loader.load()
            
            # Split into chunks
            chunks = self.text_splitter.split_documents(documents)
            
            # Add metadata to each chunk
            for chunk in chunks:
                chunk.metadata.update({
                    'version': metadata.version,
                    'date': metadata.date.isoformat(),
                    'security_tag': metadata.security_tag,
                    'access_tags': ','.join(metadata.access_tags)
                })
            
            return chunks
        except Exception as e:
            logging.error(f"Error processing file {file_path}: {str(e)}")
            return []

    def store_document(self, 
                      file_path: str, 
                      metadata: DocumentMetadata) -> bool:
        """Store a document with its metadata in the vector store."""
        try:
            chunks = self.process_document(file_path, metadata)
            if not chunks:
                return False
                
            self.vectorstore.add_documents(chunks)
            # No need to call persist() as PersistentClient handles persistence automatically
            return True
        except Exception as e:
            logging.error(f"Error storing document {file_path}: {str(e)}")
            return False

    def retrieve_by_date(self, 
                        target_date: datetime,
                        security_level: Optional[str] = None,
                        k: int = 5) -> List[Document]:
        """Retrieve document chunks closest to the given date."""
        try:
            # Build filter based on security level
            where = None
            if security_level:
                where = {"security_tag": {"$eq": security_level}}
            
            # Get all documents from the vector store
            results = self.vectorstore.similarity_search(
                query="",  # Empty query to get all documents
                k=k,
                filter=where  # Use the properly formatted where clause
            )
            
            if not results:
                return []
            
            # Sort by date proximity
            sorted_results = sorted(
                results,
                key=lambda x: abs(
                    datetime.fromisoformat(x.metadata['date']) - target_date
                )
            )[:k]
            
            return sorted_results
        except Exception as e:
            logging.error(f"Error retrieving documents by date: {str(e)}")
            return []

    def retrieve_by_version(self, 
                          version: str,
                          k: int = 5) -> List[Document]:
        """Retrieve document chunks by version tag."""
        try:
            results = self.vectorstore.similarity_search(
                "",  # Empty query to get all documents
                filter={"version": version},
                k=k
            )
            return results
        except Exception as e:
            logging.error(f"Error retrieving documents by version: {str(e)}")
            return []

    def retrieve_by_content_similarity(self,
                                     query: str,
                                     security_level: Optional[str] = None,
                                     k: int = 5) -> List[Document]:
        """Retrieve document chunks based on content similarity."""
        try:
            # Build filter based on security level
            where = None
            if security_level:
                where = {"security_tag": {"$eq": security_level}}
            
            # Search using similarity
            results = self.vectorstore.similarity_search(
                query,
                k=k,
                filter=where  # Use the properly formatted where clause
            )
            return results
        except Exception as e:
            logging.error(f"Error retrieving documents by similarity: {str(e)}")
            return []

def run_test_cases():
    """Run test cases to demonstrate functionality."""
    # Configure logging
    logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

    # Test data structure

    # Configure logging
    logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

    # Test data structure
    test_documents = [

        {   
            "PDF": "1",
            "content": "If a triangle has two equal sides, then it is an isosceles triangle.",
            "metadata": DocumentMetadata(
                version="v1.0",
                date=datetime(2023, 1, 1),
                security_tag="Public",
                access_tags=["all"]
            )
        },
        {
            "PDF": "1",
            "content": "If a triangle has two equal sides and two equal angles, then it is an equilateral triangle.",
            "metadata": DocumentMetadata(
                version="v2.0",
                date=datetime(2023, 3, 15),
                security_tag="Confidential",
                access_tags=["restricted"]
            )
        },
        {   
            "PDF": "2",
            "content": "If a triangle has three equal sides, then it is an equilateral triangle.",
            "metadata": DocumentMetadata(
                version="v1.1",
                date=datetime(2023, 2, 1),
                security_tag="Restricted",
                access_tags=["confidential"]
            )
        },
        {
            "PDF": "2",
            "content": "If a triangle has three equal sides and three equal angles, then it is a regular triangle.",
            "metadata": DocumentMetadata(
                version="v2.1",
                date=datetime(2023, 4, 1),
                security_tag="Top Secret",
                access_tags=["top-secret"]
            )
        }
    ]
    
    # Initialize processor with test directory
    processor = LangChainPDFProcessor(persist_directory="test_db")
    
    print("Creating and storing test documents...")
    
    for doc in test_documents:
        temp = None
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as temp:
                # Write content to file
                temp.write(doc["content"])
                temp.flush()
                
                # Store document with metadata
                success = processor.store_document(temp.name, doc["metadata"])
                if not success:
                    logging.error(f"Failed to store document with content: {doc['content'][:50]}...")
        except Exception as e:
            logging.error(f"Error processing test document: {str(e)}")
            print(f"Error processing test document: {str(e)}")
        finally:
            # Clean up temporary file
            if temp:
                try:
                    os.unlink(temp.name)
                except Exception as e:
                    logging.error(f"Error deleting temporary file: {str(e)}")

    # Test case 1: Retrieve by date
    print("\nTest case 1 - Retrieve by date (2023-03-08):")
    results = processor.retrieve_by_date(datetime(2023, 3, 8))

    for doc in results:
        
        print(f"Content: {doc.page_content[:120]}")
        print(f"Version: {doc.metadata['version']}")
        print(f"Date: {doc.metadata['date']}")
        print(f"Security Tag: {doc.metadata['security_tag']}")
        print()
    
    # Test case 2: Retrieve by date (2023-02-15)
    print("\nTest case 2 - Retrieve by date (2023-02-15):")
    results = processor.retrieve_by_date(datetime(2023, 2, 15))

    for doc in results:
        print(f"Content: {doc.page_content[:120]}")
        print(f"Version: {doc.metadata['version']}")
        print(f"Date: {doc.metadata['date']}")
        print(f"Security Tag: {doc.metadata['security_tag']}")
        print()
    
    # Test case 3: Retrieve by security tag
    print("\nTest case 3 - Retrieve by security tag (Confidential):")
    results = processor.retrieve_by_date(
        datetime(2023, 3, 1),
        security_level="Confidential"
    )
    for doc in results:
        print(f"Content: {doc.page_content[:120]}")
        print(f"Version: {doc.metadata['version']}")
        print(f"Date: {doc.metadata['date']}")
        print(f"Security Tag: {doc.metadata['security_tag']}")

        print()
    
    # Test case 4: Content similarity search
    print("\nTest case 4 - Content similarity search:")
    query = "triangle with equal sides"
    print(f"Query: {query}")
    results = processor.retrieve_by_content_similarity(query)
    for doc in results:
        print(f"Content: {doc.page_content[:120]}")
        print(f"Version: {doc.metadata['version']}")
        print(f"Date: {doc.metadata['date']}")
        print()


if __name__ == "__main__":
    run_test_cases()