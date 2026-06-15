# Document Loaders in RAG - Complete Blueprint & Guide

## 📋 Table of Contents
1. [Overview & Purpose](#-overview--purpose)
2. [What Are Document Loaders?](#-what-are-document-loaders)
3. [Why Document Loaders Matter in RAG](#-why-document-loaders-matter-in-rag)
4. [Core Concepts](#-core-concepts)
5. [Tech Stack & Tools](#-tech-stack--tools)
6. [Detailed Loader Implementations](#-detailed-loader-implementations)
7. [Practical Takeaways](#-practical-takeaways)
8. [Interview Questions](#-interview-questions)

---

## Overview & Purpose

This repository demonstrates **Document Loaders**, the first critical step in building a Retrieval-Augmented Generation (RAG) system. Document Loaders are Python utilities that extract text and structured data from various document formats and convert them into a standardized format that AI models can process.

### Repository Goal
- **Teach** how to load different document types (CSV, JSON, PDF, Text, Web)
- **Demonstrate** practical implementations with real-world examples
- **Provide** reusable patterns for document ingestion in RAG pipelines
- **Enable** beginners to understand the foundation of RAG systems

### What You Will Learn
✅ How to extract data from different file formats  
✅ How metadata is handled and preserved  
✅ How to parse structured and unstructured data  
✅ How to handle complex PDFs with images and tables  
✅ How to scrape web content programmatically  

---

## What Are Document Loaders?

### Definition
**Document Loaders** are specialized Python components that:
- Read files from various sources (local files, web, databases)
- Extract and parse content in a consistent format
- Preserve important metadata (source, page number, URL, etc.)
- Return standardized `Document` objects that downstream RAG components can process

### The Document Object Structure

Every loader returns a list of `Document` objects with this structure:

```python
{
    "page_content": "The actual text content extracted from the document",
    "metadata": {
        "source": "path/to/file.csv",
        "page": 1,
        "author": "John Doe",
        # ... other metadata specific to the document type
    }
}
```

**Key Components:**
- **`page_content`**: The actual text/data content extracted from the document
- **`metadata`**: Contextual information about the document (source, page number, author, etc.)

---

## Why Document Loaders Matter in RAG

### The RAG Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. DOCUMENT LOADING (This Repository) ← You are here            │
│    Extract raw data from various sources                        │
│                          ↓                                       │
│ 2. TEXT SPLITTING                                                │
│    Break large documents into manageable chunks                 │
│                          ↓                                       │
│ 3. EMBEDDINGS                                                    │
│    Convert text chunks to numerical vectors                     │
│                          ↓                                       │
│ 4. VECTOR STORE                                                  │
│    Store embeddings for fast retrieval                          │
│                          ↓                                       │
│ 5. RETRIEVAL                                                     │
│    Find relevant chunks for a user query                        │
│                          ↓                                       │
│ 6. GENERATION                                                    │
│    Feed retrieved context to LLM to generate answer             │
└─────────────────────────────────────────────────────────────────┘
```

### Why This Step Is Critical

Without proper document loading:
- ❌ Data is lost or corrupted
- ❌ Important context (metadata) is discarded
- ❌ The LLM receives incomplete or wrong information
- ❌ RAG system produces poor quality answers

With proper document loading:
- ✅ All relevant data is preserved
- ✅ Metadata allows tracking source of information
- ✅ Consistent format for downstream processing
- ✅ Better RAG system performance

---

## Core Concepts

### 1. **Content vs. Metadata**

**Content**: The actual text/data to be processed
```python
"Oversize-fit coat made of a viscose blend fabric. Notch lapel collar..."
```

**Metadata**: Information ABOUT the content
```python
{
    "source": "apparels.json",
    "product_name": "PINSTRIPE COAT",
    "category": "Men Cloths",
    "price": 4900
}
```

### 2. **Schema Mapping**

In structured data (CSV, JSON), you must tell the loader:
- Which column/field contains the actual content
- Which fields should become metadata

```python
# CSV Example
loader = CSVLoader(
    file_path="organizations.csv",
    source_column="Industry",                        # Becomes source in metadata
    metadata_columns=["Website", "Founded"],         # Become metadata
    content_columns=["Description"]                  # Becomes page_content
)
```

### 3. **Custom Metadata Functions**

For complex data structures, use custom functions to extract metadata:

```python
def metadata_func(record: dict, metadata: dict) -> dict:
    metadata["product_name"] = record["productName"]
    metadata["price"] = record["price"]
    del metadata["seq_num"]  # Remove unwanted fields
    return metadata
```

### 4. **Extraction Modes**

Different loaders support different extraction modes:
- **Page mode**: Each page becomes a separate document
- **Stream mode**: Process large files without loading everything into memory
- **Lazy loading**: Load documents on-demand

### 5. **Image Extraction from PDFs**

Modern PDFs may contain images, diagrams, and tables. Special parsers extract text from images:
- **RapidOCRBlobParser**: Fast OCR (Optical Character Recognition) engine
- **TesseractBlobParser**: Advanced OCR engine (slower but more accurate)

---

## Tech Stack & Tools

### Core Libraries

| Library | Purpose | Why It's Used |
|---------|---------|---------------|
| **LangChain** | Framework for LLM applications | Provides standardized loader interface |
| **langchain-community** | Community integrations | Contains 50+ document loader implementations |
| **PyPDF** / **PyPDF2** | PDF text extraction | Reads text from PDF documents |
| **PDFMiner** | Advanced PDF processing | Handles complex PDF layouts with images |
| **PDFPlumber** | Structured PDF extraction | Excellent for tables and structured data in PDFs |
| **BeautifulSoup4** | HTML/Web parsing | Extracts content from HTML documents |
| **RapidOCR** | Fast OCR | Extracts text from images in PDFs |
| **Tesseract** | Advanced OCR | Alternative high-accuracy OCR engine |
| **jq** | JSON query language | Parses nested JSON structures with JQ schema |

### Python Environment

```bash
# Key dependencies from requirements.txt
langchain                    # Core RAG framework
langchain-community          # Community loaders
pypdf                        # PDF parsing
rapidocr-onnxruntime         # Fast OCR for images
pytesseract                  # Advanced OCR option
pdfplumber                   # Structured PDF extraction
pdfminer.six                 # Advanced PDF processing
beautifulsoup4               # Web scraping
lxml                         # XML/HTML parsing
ipykernel                    # Jupyter kernel for notebooks
```

---

## Detailed Loader Implementations

### 1️⃣ CSV Loader

**Use Case**: Loading tabular data from CSV files (spreadsheets, data exports)

**What It Does**:
- Reads CSV file row by row
- Converts each row into a Document
- Allows you to specify which columns are content vs. metadata

#### Code Walkthrough

```python
from langchain_community.document_loaders.csv_loader import CSVLoader
from pathlib import Path

# Step 1: Define file path
file_path = Path("knowledge-source/organizations.csv")

# Step 2: Create loader with schema mapping
loader = CSVLoader(
    file_path=file_path,
    source_column="Industry",                    # This column becomes the "source"
    metadata_columns=["Website", "Founded", "Number of employees"],
    content_columns=["Description"]              # This is the main content
)

# Step 3: Load documents
documents = loader.load()

# Result: 9 documents (one per row)
# Each document has:
# - page_content: "Ergonomic zero administration knowledge user"
# - metadata: {
#     "source": "Online Publishing",
#     "Website": "http://www.day-hartman.org/",
#     "Founded": 1980,
#     "Number of employees": 6852
#   }
```

#### Key Concepts

- **Row-to-Document Mapping**: Each CSV row becomes one Document
- **Schema Configuration**: Explicitly define which columns are content and metadata
- **Metadata Enrichment**: Additional columns become metadata for tracking/filtering

#### When to Use
- ✅ Business data (employees, sales, organizations)
- ✅ Product catalogs
- ✅ Research data
- ✅ Any tabular/structured data

---

### 2️⃣ JSON Loader

**Use Case**: Loading structured data from JSON files (APIs, config files, data exports)

**What It Does**:
- Parses JSON files
- Uses JQ schema for navigation (similar to XPath for JSON)
- Applies custom metadata extraction functions
- Handles nested JSON structures

#### Code Walkthrough

```python
from langchain_community.document_loaders.json_loader import JSONLoader
from pathlib import Path

# Sample JSON structure:
# {
#   "products": [
#     {
#       "productName": "PINSTRIPE COAT",
#       "Description": "Oversize-fit coat...",
#       "price": 4900,
#       "category": "Men Cloths"
#     }
#   ]
# }

# Step 1: Define custom metadata extraction
def metadata_func(record: dict, metadata: dict) -> dict:
    # Extract fields from the record into metadata
    metadata["product_name"] = record["productName"]
    metadata["category"] = record["category"]
    metadata["price"] = record["price"]
    
    # Remove unwanted auto-generated fields
    del metadata["seq_num"]
    
    return metadata

# Step 2: Create loader with JQ schema
file_path = Path("knowledge-source/apparels.json")
loader = JSONLoader(
    file_path=file_path.as_posix(),
    jq_schema=".products[]",               # Navigate to products array
    content_key="Description",             # Use Description as page_content
    metadata_func=metadata_func            # Apply custom metadata function
)

# Step 3: Load documents
documents = loader.load()

# Result: 3 documents (one per product)
# Each document has:
# - page_content: "Oversize-fit coat made of a viscose blend fabric..."
# - metadata: {
#     "product_name": "PINSTRIPE COAT",
#     "category": "Men Cloths",
#     "price": 4900,
#     "source": "apparels.json"
#   }
```

#### Key Concepts

- **JQ Schema**: Query language for navigating JSON (`.products[]` means "iterate over products array")
- **content_key**: Which field in the JSON becomes the page_content
- **Custom Metadata Functions**: Transform raw record data into metadata
- **Nested Data Handling**: Automatically extracts arrays from nested JSON

#### Common JQ Patterns

```python
".products[]"           # Iterate over products array
".data.items[]"         # Navigate nested structure
".[0]"                  # Get first element
".[] | select(.type == 'product')"  # Filter elements
```

#### When to Use
- ✅ API responses
- ✅ Configuration files
- ✅ Product/service data
- ✅ Any nested hierarchical data

---

### 3️⃣ Text Loader

**Use Case**: Loading plain text files (notes, articles, reports, TXT files)

**What It Does**:
- Reads plain text files
- Preserves line breaks and formatting
- Extracts file metadata (source path, encoding)

#### Code Walkthrough

```python
from langchain_community.document_loaders import TextLoader
from pathlib import Path

# Step 1: Define file path
file_path = Path("knowledge-source/transformers.txt")

# Step 2: Create loader (very simple, minimal configuration)
loader = TextLoader(file_path=file_path)

# Step 3: Load documents
documents = loader.load()

# Result: 1 document (entire file is one document)
# document has:
# - page_content: """# Transformer Model in Large Language Models (LLMs)
#                   
#                   This note explains the Transformer model...
#                   (entire file content)"""
# - metadata: {
#     "source": "../knowledge-source/transformers.txt"
#   }
```

#### Key Concepts

- **Single Document Output**: Entire file becomes one Document (unlike CSV/JSON where each row/item becomes a document)
- **Minimal Configuration**: No schema mapping needed
- **Whitespace Preservation**: Maintains original formatting and line breaks
- **Metadata Simplicity**: Only metadata is the file source

#### Output Structure Comparison

```python
# CSV/JSON: Multiple documents (one per row/item)
len(documents)  # Example: 9 documents for 9 rows

# Text: Single document (entire file)
len(documents)  # Result: 1 document
```

#### When to Use
- ✅ Plain text files
- ✅ Notes and documentation
- ✅ Articles and blog posts
- ✅ Code files
- ✅ Any unstructured text content

---

### 4️⃣ PDF Loader (3 Different Approaches)

**Use Case**: Loading documents from PDF files (research papers, reports, books)

**Why Multiple Approaches?**
Different PDF structures require different extraction techniques:

#### Approach 1: PyPDFLoader (Basic)

**Best For**: Simple PDFs with mostly text

```python
from langchain_community.document_loaders.pdf import PyPDFLoader
from pathlib import Path

# Step 1: Define file path
file_path = Path("knowledge-source/attention_is_all_you_need.pdf")

# Step 2: Create loader
loader = PyPDFLoader(
    file_path=file_path.as_posix(),
    mode='page'              # One document per page
)

# Step 3: Load documents
documents = loader.load()

# Result: 16 documents (one per page)
# Each document has:
# - page_content: "Attention Is All You Need\n\nAshish Vaswani..."
# - metadata: {
#     "source": "attention_is_all_you_need.pdf",
#     "page": 0
#   }
```

**Advantages**:
- Simple and fast
- Good for text-heavy PDFs
- Lightweight

**Disadvantages**:
- Doesn't extract images
- May struggle with complex layouts
- Tables might be formatted poorly

---

#### Approach 2: PyPDFLoader with Image Extraction

**Best For**: PDFs with images, diagrams, tables

```python
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.parsers import RapidOCRBlobParser

# Step 1: Define file path
file_path = Path("knowledge-source/attention_is_all_you_need.pdf")

# Step 2: Create loader WITH image extraction
loader = PyPDFLoader(
    file_path=file_path.as_posix(),
    mode="page",
    extract_images=True,                        # Enable image extraction
    images_parser=RapidOCRBlobParser(),         # Use RapidOCR for OCR
    images_inner_format="html-img"              # Embed images as HTML
)

# Step 3: Load documents
documents = loader.load()

# Result: 16 documents, each containing:
# - Text extracted from the page
# - Text extracted FROM images using OCR
# - Embedded as: "<!-- image --> <html><img src='...'></html>"

# The page_content now includes both text AND text from images!
```

**Key Concepts**:
- **OCR (Optical Character Recognition)**: Converts images to text
- **RapidOCRBlobParser**: Fast OCR engine, good for real-time processing
- **HTML Embedding**: Images embedded as HTML img tags for context

---

#### Approach 3: PDFMinerLoader (Advanced)

**Best For**: Complex PDFs with tables, multi-column layouts, scanned documents

```python
from langchain_community.document_loaders import PDFMinerLoader
from langchain_community.document_loaders.parsers import RapidOCRBlobParser

# Step 1: Define file path
file_path = Path("knowledge-source/attention_is_all_you_need.pdf")

# Step 2: Create advanced loader
loader = PDFMinerLoader(
    file_path=file_path.as_posix(),
    mode="page",
    extract_images=True,
    images_parser=RapidOCRBlobParser(),
    images_inner_format="html-img"
)

# Step 3: Load documents
documents_with_images = loader.load()

# Result: Superior extraction of:
# - Complex table layouts
# - Multi-column text
# - Scanned documents (via OCR)
# - Mathematical equations (as images)

# Example: Extracting page 5 content
for doc in documents_with_images:
    if doc.metadata["page"] == 5:
        print(doc.page_content)  # Includes tables + text + image text
        break
```

**Advantages**:
- Handles complex layouts
- Better table extraction
- Excellent for scanned PDFs
- Most accurate content extraction

---

#### Approach 4: PDFPlumberLoader (Best for Structured Data)

**Best For**: PDFs with tables, forms, and structured data

```python
from langchain_community.document_loaders import PDFPlumberLoader

# Step 1: Define file path
file_path = Path("knowledge-source/attention_is_all_you_need.pdf")

# Step 2: Create loader
loader = PDFPlumberLoader(file_path=file_path.as_posix())

# Step 3: Load documents
documents = loader.load()

# Result: Excellent for:
# - Table extraction as HTML/markdown
# - Form fields
# - Structured data in PDFs
```

**Advantages**:
- Specialized for tables and structured data
- Can extract tables as HTML/markdown
- Fast and accurate for form data

---

#### PDF Loader Comparison

| Loader | Text Quality | Tables | Images | Speed | Best Use |
|--------|---|---|---|---|---|
| **PyPDFLoader** | ⭐⭐⭐ | ⭐ | ❌ | ⭐⭐⭐ | Simple PDFs |
| **PyPDFLoader + OCR** | ⭐⭐⭐ | ⭐⭐ | ✅ | ⭐⭐ | Mixed content |
| **PDFMinerLoader** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ✅ | ⭐ | Complex PDFs |
| **PDFPlumberLoader** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ❌ | ⭐⭐ | Tables/Forms |

---

### 5️⃣ Web Loader (2 Approaches)

**Use Case**: Scraping and loading content from websites

**What It Does**:
- Fetches HTML from URLs
- Parses HTML to extract readable text
- Preserves metadata (URL, title, etc.)

#### Approach 1: WebBaseLoader (Static URLs)

**Best For**: Loading specific web pages

```python
from langchain_community.document_loaders import WebBaseLoader
from pprint import pp

# Step 1: Define URLs to load
url_1 = "https://docs.langchain.com/oss/python/integrations/document_loaders/pypdfloader"
url_2 = "https://docs.langchain.com/oss/python/integrations/document_loaders/pdfminer"
url_3 = "https://docs.langchain.com/oss/python/integrations/document_loaders/pdfplumber"

# Step 2: Create loader
loader = WebBaseLoader(web_paths=[url_1, url_2, url_3])

# Step 3: Load documents
documents = loader.load()

# Result: 3 documents (one per URL)
# Each document has:
# - page_content: "PyPDFLoader\n\nPyPDF is a pure python PDF library..."
# - metadata: {
#     "source": "https://docs.langchain.com/oss/python/...",
#     "title": "PyPDFLoader",
#     "description": "..."
#   }

# Example: Access specific document
print(documents[0].page_content)     # Extracted text
pp(documents[0].metadata)            # Source URL, title, etc.
```

**Key Features**:
- Clean HTML parsing
- Automatic metadata extraction
- Handles multiple URLs
- Returns readable text (removes HTML tags)

---

#### Approach 2: RecursiveUrlLoader (Website Crawling)

**Best For**: Loading entire sections of websites (with depth limits)

```python
from langchain_community.document_loaders import RecursiveUrlLoader

# Step 1: Define base URL and depth
base_url = "https://docs.langchain.com/oss/python/integrations/document_loaders"

# Step 2: Create loader with depth limit
loader = RecursiveUrlLoader(
    url=base_url,
    max_depth=2        # Crawl up to 2 levels deep
)

# Step 3: Load documents
documents = loader.load()

# Result: Dozens of documents (all pages reachable from base URL within 2 levels)

# Example: Find total number of pages loaded
len(documents)  # Result: 45+ pages

# Option 1: Load everything at once
documents = loader.load()

# Option 2: Lazy load for memory efficiency
documents_lazy_load = loader.lazy_load()

counter = 0
for document in documents_lazy_load:
    if counter == 20:
        break
    
    print(document.page_content[0:300])
    pp(document.metadata)
    counter += 1
```

**Key Concepts**:
- **Depth Control**: `max_depth` parameter prevents infinite crawling
  - `max_depth=1`: Only immediate child pages
  - `max_depth=2`: Children and grandchildren
  - `max_depth=3`: Three levels deep
- **Lazy Loading**: Process pages one at a time without loading all into memory
- **Automatic Link Following**: Finds and follows all links on each page

**Lazy Loading Pattern**:
```python
# Memory efficient for large websites
documents_lazy = loader.lazy_load()

for doc in documents_lazy:
    process_document(doc)  # Process each document as it's loaded
```

---

### Web Loader Comparison

| Loader | Use Case | Speed | Memory | Best For |
|--------|----------|-------|--------|----------|
| **WebBaseLoader** | Specific URLs | ⭐⭐⭐ | ⭐⭐⭐ | Individual pages |
| **RecursiveUrlLoader** | Website crawling | ⭐⭐ | ⭐ | Entire sections |

---

## Practical Takeaways

### Key Learning Points from This Repository

#### 1. **Understanding Document Objects**
```python
# Every loader returns this structure
document = {
    "page_content": "actual text content",
    "metadata": {"source": "...", "other": "info"}
}

# Access in code:
text = document.page_content
source = document.metadata["source"]
```

#### 2. **Schema Design is Important**
Before choosing a loader, ask:
- ❓ What is my data format? (CSV/JSON/PDF/Text/Web)
- ❓ What content do I need? (Which field/column?)
- ❓ What metadata is important? (For filtering/tracking)
- ❓ Is there any transformation needed? (Custom functions)

#### 3. **Metadata Enables Retrieval Traceability**
Good metadata means:
- ✅ You know where each fact came from
- ✅ You can trace back to source
- ✅ You can filter results by source
- ✅ You can validate information

```python
# Bad: No metadata
"The company was founded in 1980"  # Where did this come from?

# Good: With metadata
page_content: "The company was founded in 1980"
metadata: {
    "source": "organizations.csv",
    "organization": "Liu-Hoover",
    "row": 1
}
```

#### 4. **Choose the Right Loader for Your Data**

| Data Type | Loader | Why |
|-----------|--------|-----|
| Structured data (rows/columns) | **CSVLoader** | Natural row-to-document mapping |
| Hierarchical data (nested) | **JSONLoader** | JQ schema for navigation |
| Plain text | **TextLoader** | Simplest, minimal overhead |
| Scientific papers, reports | **PyPDFLoader** | Text-focused extraction |
| Complex layouts, tables | **PDFMinerLoader** | Advanced layout understanding |
| Tables, forms | **PDFPlumberLoader** | Specialized for structured PDFs |
| Website pages | **WebBaseLoader** | Single/multiple specific URLs |
| Website sections | **RecursiveUrlLoader** | Crawling with depth control |

#### 5. **PDF Loaders: Understanding the Trade-offs**

```python
# Fast but basic
PyPDFLoader()

# Medium complexity, handles images
PyPDFLoader(extract_images=True, images_parser=RapidOCRBlobParser())

# Complex but comprehensive
PDFMinerLoader(extract_images=True, images_parser=RapidOCRBlobParser())

# Best for tables
PDFPlumberLoader()
```

#### 6. **Memory Efficiency with Lazy Loading**
```python
# Load everything at once (high memory for large datasets)
documents = loader.load()

# Load on-demand (efficient for web crawling)
for doc in loader.lazy_load():
    process(doc)  # Process one at a time
```

#### 7. **Metadata Functions Allow Custom Transformation**
```python
# Transform raw data into useful metadata
def metadata_func(record, metadata):
    metadata["product_id"] = record["id"]
    metadata["price_usd"] = record["price"]
    del metadata["internal_seq"]  # Remove unnecessary fields
    return metadata
```

---

## Interview Questions

### Beginner Level

**Q1: What is a Document Loader and why is it the first step in RAG?**

**A:** A Document Loader is a Python component that reads documents from various sources and converts them into standardized `Document` objects containing text content and metadata. It's the first step in RAG because RAG systems need clean, structured data to work with. Without proper document loading:
- Data might be corrupted or incomplete
- Important context is lost
- Downstream components (embeddings, retrieval) receive poor quality input
- Final answer quality suffers

---

**Q2: What are the two main parts of a Document object returned by loaders?**

**A:** 
1. **page_content**: The actual extracted text/data from the document
2. **metadata**: Contextual information about the document (source, page number, date, etc.)

Example:
```python
Document(
    page_content="The company manufactures electronics",
    metadata={"source": "data.csv", "industry": "Manufacturing"}
)
```

---

**Q3: Why do we need different loaders for different file formats?**

**A:** Different file formats have different structures:
- **CSV**: Tabular with rows and columns - best to create one document per row
- **JSON**: Hierarchical/nested - need JQ schema to navigate
- **PDF**: Can have text, images, tables - need specialized parsers
- **Text**: Plain text - simple, one document per file
- **Web**: HTML with links - need to parse and follow links

Each loader understands the specific structure and extracts data efficiently.

---

**Q4: In the CSV loader example, what does `source_column` do?**

**A:** The `source_column` parameter specifies which CSV column becomes the `source` field in the document's metadata. This helps track where each document came from.

```python
loader = CSVLoader(
    file_path="organizations.csv",
    source_column="Industry"  # "Industry" column values become metadata source
)
# If a row has Industry="Online Publishing", 
# the document's metadata["source"] = "Online Publishing"
```

---

**Q5: What is JQ schema in JSONLoader and why is it useful?**

**A:** JQ is a query language for navigating JSON structures, similar to XPath for XML. It's useful because:
- JSON files can have nested structures
- You need to tell the loader which part of the JSON to extract
- It allows filtering and transforming data

Example:
```python
jq_schema=".products[]"  # Navigate to products array
jq_schema=".data.items[] | select(.active==true)"  # Filter active items
```

---

**Q6: What's the difference between `PyPDFLoader` and `PDFMinerLoader`?**

**A:**
- **PyPDFLoader**: Simple, fast, good for text-heavy PDFs
- **PDFMinerLoader**: More advanced, better handles complex layouts, tables, images

Choose PDFMiner when dealing with scientific papers, forms, or PDFs with complex layouts. Choose PyPDF for simple, text-only documents.

---

**Q7: What is OCR and when do we need it?**

**A:** OCR (Optical Character Recognition) converts images to text. We need it when:
- PDFs contain scanned documents
- PDFs have diagrams or charts with text
- PDFs are images of text (not digital text)

In the code:
```python
from langchain_community.document_loaders.parsers import RapidOCRBlobParser

loader = PyPDFLoader(
    extract_images=True,
    images_parser=RapidOCRBlobParser()  # Enable OCR
)
```

---

### Intermediate Level

**Q8: Explain the metadata_func in JSONLoader. Why do we need it?**

**A:** `metadata_func` is a custom function that transforms raw record fields into metadata. We need it because:
- Raw data might have unnecessary fields
- Field names might not be appropriate for metadata
- We need to select/rename specific fields for context

Example:
```python
def metadata_func(record: dict, metadata: dict) -> dict:
    metadata["product_name"] = record["productName"]
    metadata["price"] = record["price"]
    del metadata["seq_num"]  # Remove auto-generated field
    return metadata
```

This transforms raw record data into clean, meaningful metadata.

---

**Q9: In the WebBaseLoader example, we load 3 specific URLs. How is this different from RecursiveUrlLoader?**

**A:**
- **WebBaseLoader**: Takes a list of specific URLs and loads each one. No automatic link following.
- **RecursiveUrlLoader**: Takes a base URL and automatically finds and follows links up to a specified depth.

```python
# WebBaseLoader - explicitly list URLs
WebBaseLoader(web_paths=["url1", "url2", "url3"])

# RecursiveUrlLoader - automatic crawling
RecursiveUrlLoader(url="base_url", max_depth=2)  # Crawls automatically
```

---

**Q10: What is lazy_load() and when should we use it?**

**A:** `lazy_load()` returns a generator that loads documents one at a time, instead of loading everything into memory at once. Use it when:
- Loading large websites (hundreds of pages)
- Memory is limited
- You want to process documents as they arrive

```python
# Load all at once (high memory)
documents = loader.load()

# Load one at a time (memory efficient)
for doc in loader.lazy_load():
    process(doc)
```

---

**Q11: How would you choose between TextLoader and CSVLoader for a data source?**

**A:**
- **TextLoader**: Use when you have plain text files where the entire content is one document
- **CSVLoader**: Use when you have tabular data where each row should be a separate document with specific fields

```python
# Text file - one document
TextLoader("document.txt")
# Result: 1 document containing entire file

# CSV file - multiple documents
CSVLoader("data.csv")
# Result: N documents (one per row)
```

---

**Q12: In the PDF loading code, what does `mode='page'` do and what are the alternatives?**

**A:** `mode='page'` means each page of the PDF becomes a separate document. This is useful for:
- Tracking which page content came from
- Keeping document size manageable
- Maintaining page-level structure

Alternative modes:
- `mode='page'`: One document per page (most common)
- Other modes: Depends on the specific loader (some don't have alternatives)

---

### Advanced Level

**Q13: Design a RAG system to load a company's documentation (PDFs, web pages, and internal data). What loaders would you use and why?**

**A:** This requires multiple loaders:

```
Company Documentation RAG System:
├── PDFs (research papers, reports)
│   └── PDFMinerLoader  (complex layouts, images, tables)
├── Website (public docs)
│   └── RecursiveUrlLoader  (crawl documentation site)
├── CSV/Database exports (structured data)
│   └── CSVLoader  (tabular data)
└── Internal notes (plain text)
    └── TextLoader  (simple text files)

Rationale:
- PDFMinerLoader: Company reports often have complex formatting
- RecursiveUrlLoader: Documentation sites have interconnected pages
- CSVLoader: Structured company data needs row-by-row extraction
- TextLoader: Internal notes are usually plain text
```

---

**Q14: A PDF contains both text and scanned images of text. Which loader would you use and how would you configure it?**

**A:** Use PDFMinerLoader with OCR enabled:

```python
from langchain_community.document_loaders import PDFMinerLoader
from langchain_community.document_loaders.parsers import RapidOCRBlobParser

loader = PDFMinerLoader(
    file_path="mixed_content.pdf",
    mode="page",
    extract_images=True,                  # Enable image extraction
    images_parser=RapidOCRBlobParser(),  # Use fast OCR
    images_inner_format="html-img"        # Embed images as HTML
)

documents = loader.load()
# Result: Documents containing both digital text AND extracted image text
```

---

**Q15: Explain the trade-off between PyPDFLoader, PDFMinerLoader, and PDFPlumberLoader. When would you use each?**

**A:**

```
PyPDFLoader:
- Pros: Fast, simple, minimal overhead
- Cons: Poor at complex layouts, no images
- Use: Simple text-only PDFs

PDFMinerLoader:
- Pros: Excellent layout handling, OCR support, tables
- Cons: Slower, more complex
- Use: Complex research papers, scanned documents

PDFPlumberLoader:
- Pros: Excellent for structured data (tables, forms)
- Cons: No image extraction, specialized use case
- Use: PDFs containing tables or forms

Decision Matrix:
- PDF is text-only and simple? → PyPDFLoader
- PDF has complex layout or images? → PDFMinerLoader
- PDF is mostly tables/forms? → PDFPlumberLoader
- Unsure? → Start with PDFMinerLoader (most robust)
```

---

**Q16: How would you handle a very large website with thousands of pages using RecursiveUrlLoader?**

**A:** Use lazy loading with careful depth and filtering:

```python
from langchain_community.document_loaders import RecursiveUrlLoader

loader = RecursiveUrlLoader(
    url="https://example.com/docs",
    max_depth=2  # Limit depth to prevent explosion
)

# Use lazy loading for memory efficiency
processed_count = 0
for document in loader.lazy_load():
    if processed_count > 5000:  # Stop after 5000 pages
        break
    
    # Process one document at a time
    process_document(document)
    processed_count += 1

# Alternative: Use filtering
def should_include(url):
    return "docs" in url and not url.endswith(".pdf")

loader = RecursiveUrlLoader(
    url="base",
    max_depth=2,
    is_valid_url=should_include  # Skip certain URLs
)
```

---

**Q17: You have a nested JSON file with variable schema (different documents have different fields). How would you handle this?**

**A:** Use a robust metadata function that handles optional fields:

```python
def flexible_metadata_func(record: dict, metadata: dict) -> dict:
    # Handle optional fields gracefully
    metadata["title"] = record.get("title", "Unknown")
    metadata["author"] = record.get("author", "Anonymous")
    metadata["date"] = record.get("publish_date", record.get("created_date", None))
    
    # Only add if exists
    if "category" in record:
        metadata["category"] = record["category"]
    
    # Remove default fields
    if "seq_num" in metadata:
        del metadata["seq_num"]
    
    return metadata

loader = JSONLoader(
    file_path="variable_schema.json",
    jq_schema=".items[]",
    content_key="description",
    metadata_func=flexible_metadata_func
)
```

---

**Q18: How would you combine multiple loaders (CSV + PDF + Web) into a single unified RAG system?**

**A:**

```python
from langchain_community.document_loaders import (
    CSVLoader, PDFMinerLoader, WebBaseLoader
)
from langchain_community.document_loaders.parsers import RapidOCRBlobParser

# Load from different sources
csv_documents = CSVLoader(file_path="data.csv").load()

pdf_documents = PDFMinerLoader(
    file_path="report.pdf",
    extract_images=True,
    images_parser=RapidOCRBlobParser()
).load()

web_documents = WebBaseLoader(
    web_paths=["https://example.com/page1", "https://example.com/page2"]
).load()

# Combine all documents
all_documents = csv_documents + pdf_documents + web_documents

# Now all documents have:
# - Consistent page_content format
# - Consistent metadata structure
# - source field tracking origin

# Use in RAG pipeline
for doc in all_documents:
    print(f"Content from {doc.metadata['source']}: {doc.page_content[:100]}")
```

---

**Q19: What could go wrong with document loading and how would you handle errors?**

**A:**

```python
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def safe_load_documents(loader, source_name):
    try:
        documents = loader.load()
        logger.info(f"Successfully loaded {len(documents)} documents from {source_name}")
        return documents
    except FileNotFoundError:
        logger.error(f"File not found for {source_name}")
        return []
    except Exception as e:
        logger.error(f"Error loading {source_name}: {e}")
        return []

# Usage
csv_docs = safe_load_documents(
    CSVLoader(file_path="data.csv"),
    "CSV data"
)

pdf_docs = safe_load_documents(
    PDFMinerLoader(file_path="document.pdf"),
    "PDF document"
)

web_docs = safe_load_documents(
    WebBaseLoader(web_paths=["https://example.com"]),
    "Web content"
)

# Combine successfully loaded documents
all_docs = csv_docs + pdf_docs + web_docs
```

Common errors:
- FileNotFoundError: Check file paths
- JSON parsing errors: Validate JQ schema
- Network errors: Add retry logic for web loaders
- Encoding issues: Specify encoding in TextLoader
- Memory errors: Use lazy_load() for large datasets

---

**Q20: Design the architecture for a RAG system that needs to handle multiple document types with custom processing for each. What would be your approach?**

**A:**

```python
from abc import ABC, abstractmethod
from pathlib import Path

# Define abstract processor
class DocumentProcessor(ABC):
    @abstractmethod
    def load(self) -> list:
        pass
    
    @abstractmethod
    def validate(self, documents: list) -> bool:
        pass

# Implement specific processors
class CSVProcessor(DocumentProcessor):
    def load(self):
        loader = CSVLoader(
            file_path=self.file_path,
            source_column="Industry",
            metadata_columns=["Website"],
            content_columns=["Description"]
        )
        return loader.load()
    
    def validate(self, documents):
        return all(doc.page_content and doc.metadata.get("source") for doc in documents)

class PDFProcessor(DocumentProcessor):
    def load(self):
        loader = PDFMinerLoader(
            file_path=self.file_path,
            extract_images=True,
            images_parser=RapidOCRBlobParser()
        )
        return loader.load()
    
    def validate(self, documents):
        return all(doc.page_content for doc in documents)

class WebProcessor(DocumentProcessor):
    def load(self):
        loader = RecursiveUrlLoader(
            url=self.base_url,
            max_depth=2
        )
        return list(loader.lazy_load())
    
    def validate(self, documents):
        return all(doc.metadata.get("source") for doc in documents)

# Unified loader
class UnifiedDocumentLoader:
    def __init__(self):
        self.processors = {
            "csv": CSVProcessor,
            "pdf": PDFProcessor,
            "web": WebProcessor
        }
        self.documents = []
    
    def load(self, source_type, **kwargs):
        processor_class = self.processors[source_type]
        processor = processor_class(**kwargs)
        
        documents = processor.load()
        
        if processor.validate(documents):
            self.documents.extend(documents)
            logger.info(f"Loaded {len(documents)} documents from {source_type}")
        else:
            logger.error(f"Validation failed for {source_type}")
    
    def get_all_documents(self):
        return self.documents

# Usage
loader = UnifiedDocumentLoader()

loader.load("csv", file_path="companies.csv")
loader.load("pdf", file_path="research.pdf")
loader.load("web", base_url="https://docs.example.com")

all_docs = loader.get_all_documents()
# Ready for downstream RAG processing
```

This architecture:
- ✅ Handles multiple document types
- ✅ Allows custom processing per type
- ✅ Provides validation layer
- ✅ Easy to extend with new document types
- ✅ Centralized error handling

---

## Summary: What You Should Know

### Core Concepts Mastered
- ✅ Document Loaders extract data and convert to standardized format
- ✅ Every Document has `page_content` and `metadata`
- ✅ Different formats require different loaders
- ✅ Metadata preserves important context and source tracking
- ✅ PDF is complex - multiple approaches exist (PyPDF, PDFMiner, PDFPlumber)
- ✅ Web scraping can be static (WebBaseLoader) or recursive (RecursiveUrlLoader)
- ✅ Memory efficiency matters (lazy_load() for large datasets)

### Practical Skills Acquired
- ✅ Can load CSV, JSON, Text, PDF, and Web content
- ✅ Can design custom metadata extraction
- ✅ Can choose appropriate loader for any data source
- ✅ Can handle complex PDFs with images and tables
- ✅ Can combine multiple loaders for unified RAG systems
- ✅ Can optimize for memory and performance

### Next Steps in RAG Pipeline
After Document Loading, the RAG pipeline continues with:
1. **Text Splitting** (Break large documents into chunks)
2. **Embeddings** (Convert text to vectors)
3. **Vector Store** (Store embeddings for retrieval)
4. **Retrieval** (Find relevant documents)
5. **Generation** (Use LLM to answer)

---

## Resources for Further Learning

### Official Documentation
- [LangChain Document Loaders](https://docs.langchain.com/docs/integrations/document_loaders/)
- [PyPDF Documentation](https://pypdf.readthedocs.io/)
- [PDFPlumber Documentation](https://github.com/jsvine/pdfplumber)

### Related Modules in This Repository
- `2_Text_Splitters/` - Break documents into manageable chunks
- `3_Embedding_Models/` - Convert text to embeddings
- `4_Vector_Stores/` - Store and retrieve embeddings
- `5_Retrievers/` - Advanced retrieval techniques
- `6_Advanced_Retrievers/` - Specialized retrieval methods

---

## Getting Started

### Installation
```bash
pip install -r requirements.txt
```

### Running Notebooks
Each notebook is self-contained and demonstrates a specific loader:
- `notebooks/csv_loader.ipynb` - Load tabular data
- `notebooks/json_loader.ipynb` - Load hierarchical data
- `notebooks/text_loader.ipynb` - Load plain text
- `notebooks/pdf_loader.ipynb` - Load PDFs (3 approaches)
- `notebooks/web_loader.ipynb` - Load web content (2 approaches)

### Sample Data
All notebooks use sample data in `knowledge-source/`:
- `organizations.csv` - Company data
- `apparels.json` - Product catalog
- `transformers.txt` - Educational text
- `attention_is_all_you_need.pdf` - Research paper

---

**Remember**: Good document loading is the foundation of a strong RAG system. Garbage in = garbage out. Invest time in proper data loading and extraction! 🚀
