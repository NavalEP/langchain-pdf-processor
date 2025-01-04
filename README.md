# LangChain PDF Processor

A sophisticated document processing system built with LangChain and ChromaDB for handling text documents with metadata, versioning, and security tags.

## Features

- Document processing with metadata support
- Version control tracking
- Security level tagging
- Date-based retrieval
- Content similarity search
- Persistent vector storage using ChromaDB

## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/NavalEP/langchain-pdf-processor.git
cd langchain-pdf-processor
```

2. Install dependencies:
```bash
pip3 install -r requirements.txt
```

## Dependencies

- langchain
- chromadb
- huggingface-hub
- sentence-transformers


### DocumentMetadata
- version: Document version string
- date: Creation/modification datetime
- security_tag: Document security level
- access_tags: List of access permissions

### LangChainPDFProcessor
Main class providing:
- Document processing and storage
- Retrieval by date
- Retrieval by version
- Content similarity search
- Security-filtered searches

## Usage Example

```python
processor = LangChainPDFProcessor()

# Store document
metadata = DocumentMetadata(
    version="v1.0",
    date=datetime.now(),
    security_tag="Public",
    access_tags=["all"]
)
processor.store_document("document.txt", metadata)

# Retrieve documents
results = processor.retrieve_by_content_similarity("search query")
```

## Running the Program

Execute the main script:
```bash
python3 main.py
```

Sample output:
```
Creating and storing test documents...

Test case 1 - Retrieve by date (2023-03-08):
Content: If a triangle has two equal sides and two equal angles, then it is an equilateral triangle.
Version: v2.0
Date: 2023-03-15T00:00:00
Security Tag: Confidential

Content: If a triangle has three equal sides and three equal angles, then it is a regular triangle.       
Version: v2.1
Date: 2023-04-01T00:00:00
Security Tag: Top Secret

Content: If a triangle has three equal sides, then it is an equilateral triangle.
Version: v1.1
Date: 2023-02-01T00:00:00
Security Tag: Restricted

Content: If a triangle has two equal sides, then it is an isosceles triangle.
Version: v1.0
Date: 2023-01-01T00:00:00
Security Tag: Public


Test case 2 - Retrieve by date (2023-02-15):
Content: If a triangle has three equal sides, then it is an equilateral triangle.
Version: v1.1
Date: 2023-02-01T00:00:00
Security Tag: Restricted

Content: If a triangle has two equal sides and two equal angles, then it is an equilateral triangle.      
Version: v2.0
Date: 2023-03-15T00:00:00
Security Tag: Confidential

Content: If a triangle has three equal sides and three equal angles, then it is a regular triangle.       
Version: v2.1
Date: 2023-04-01T00:00:00
Security Tag: Top Secret

Content: If a triangle has two equal sides, then it is an isosceles triangle.
Version: v1.0
Date: 2023-01-01T00:00:00
Security Tag: Public


Test case 3 - Retrieve by security tag (Confidential):
Content: If a triangle has two equal sides and two equal angles, then it is an equilateral triangle.
Version: v2.0
Date: 2023-03-15T00:00:00
Security Tag: Confidential


Test case 4 - Content similarity search:
Query: triangle with equal sides
Content: If a triangle has three equal sides, then it is an equilateral triangle.
Version: v1.1
Date: 2023-02-01T00:00:00

Content: If a triangle has two equal sides and two equal angles, then it is an equilateral triangle.      
Version: v2.0
Date: 2023-03-15T00:00:00

Content: If a triangle has two equal sides, then it is an isosceles triangle.
Version: v1.0
Date: 2023-01-01T00:00:00

Content: If a triangle has three equal sides and three equal angles, then it is a regular triangle.       
Version: v2.1
Date: 2023-04-01T00:00:00
```

## Security Levels

- Public
- Confidential
- Restricted
- Top Secret
