# RAG Fusion

## Overview

`8_Rag_fusion` is a focused repository that demonstrates how to build advanced Retrieval-Augmented Generation (RAG) workflows using multiple retrieval strategies and rank fusion.

This repo is designed as a practical blueprint for creating a robust RAG pipeline with two main approaches:

- **Sub-query based retrieval fusion**: generate multiple query variations from the main question, retrieve documents for each, and fuse them.
- **Ensemble retriever fusion**: use two different retriever strategies on the same vector store and merge the results.

The goal is to show how to improve retrieval quality, reduce redundancy, and produce grounded answers from documents.

---

## What This Repo Demonstrates

This repo teaches and demonstrates:

- How to load PDF documents into a RAG pipeline using `PyPDFLoader`
- Why and how to split large documents into overlapping chunks
- How to generate embeddings using OpenAI embeddings
- How to store embeddings in a Chroma vector store
- How to build different retriever strategies
- How to merge retrieval results using Reciprocal Rank Fusion (RRF)
- How to use both LLM-driven sub-query generation and retriever ensembles
- How to ground LLM answers by injecting retrieved context into prompt templates

---

## Repo Structure

- `rag_fusion.py`
  - Implements `RAGFusion`, a reusable class for combining retrieval results.
  - Provides two factory methods:
    - `from_llm(...)` for sub-query generation-based fusion.
    - `from_retrievers(...)` for ensemble retriever-based fusion.
  - Contains a simple RRF implementation to score and rank documents.

- `main.py`
  - Minimal entrypoint placeholder.

- `rag_pipeline.ipynb`
  - Notebook demonstrating the LLM sub-query fusion workflow.
  - Shows how one query becomes many sub-queries, how each retrieves documents, and how fusion creates better context.

- `rag_pipeline_ensemble.ipynb`
  - Notebook demonstrating fusion with two retrievers using LangChain's `EnsembleRetriever`.
  - Uses similarity search and MMR search together.

- `requirements.txt` and `pyproject.toml`
  - Define dependencies needed to run the notebooks and code.

- `notebooklm_rag.pdf`
  - Example source document used by both notebooks.

---

## Core Concepts Explained

### Retrieval-Augmented Generation (RAG)

RAG is a pattern where a language model answers a question using information retrieved from external documents.

Instead of relying only on the model's internal knowledge, RAG fetches documents, extracts relevant text, and uses it as grounding for generation.

### Why Chunk Documents?

Large documents are split into smaller chunks because:

- queries can focus on smaller, more relevant passages
- vector search works better with shorter embeddings
- it reduces the risk of returning unrelated or overly broad content

### Embeddings and Vector Stores

A document chunk is converted into a dense vector using an embedding model.
Those vectors are stored in a vector database (`Chroma`), enabling semantic similarity search.

### Retriever Strategies

This repo uses two strategies:

- **Similarity Search**
  - Returns chunks most similar to the query based on vector similarity.

- **MMR (Maximal Marginal Relevance)**
  - Balances relevance and diversity.
  - Helps avoid retrieving near-duplicate chunks.
  - Controlled by `lambda_mult`.

### Ensemble Retriever

An ensemble retriever runs multiple retrieval strategies on the same data and combines their ranked results.
This is a classic fusion technique for robust retrieval.

### Sub-query Fusion

Large or complex questions are decomposed into smaller sub-queries.
Each sub-query retrieves documents independently, and the results are merged. This helps capture multiple facets of the question.

### Reciprocal Rank Fusion (RRF)

RRF is a method to combine ranked lists from different retrieval sources.
It gives higher score to documents that appear near the top in multiple ranked lists.

The formula used in this repo is:

- `score += 1 / (rank + 60)`

This encourages documents that are consistently ranked well across sub-queries or retrievers.

### Prompt Grounding & Answer Generation

The retrieved documents are joined into a single context block.
That context is passed to an LLM prompt that explicitly instructs the model to answer only from the provided text.
This reduces hallucination and improves factual accuracy.

---

## What We Coded in the Notebooks

### `rag_pipeline.ipynb`

This notebook implements a complete pipeline:

1. Load `notebooklm_rag.pdf` using `PyPDFLoader`
2. Split pages into overlapping chunks with `RecursiveCharacterTextSplitter`
3. Create embeddings using `OpenAIEmbeddings(model="text-embedding-3-small")`
4. Store chunks in `Chroma`
5. Build a similarity retriever with `k=3`
6. Use `RAGFusion.from_llm(...)` to generate 2 sub-queries from the main query
7. Retrieve documents for each sub-query and fuse them with RRF
8. Concatenate fused chunks into a grounded context
9. Generate an answer with `ChatOpenAI(model="gpt-5-mini")`

This demonstrates how LLM-powered query decomposition can improve retrieval performance.

### `rag_pipeline_ensemble.ipynb`

This notebook implements a different fusion strategy:

1. Load the same PDF and split into chunks
2. Embed and store chunks in the same Chroma vector store
3. Create two retrievers from the same store:
   - similarity search
   - MMR search
4. Build `RAGFusion.from_retrievers(...)` using `EnsembleRetriever`
5. Retrieve documents via both retrievers and fuse the results
6. Build context and ask the LLM for an answer

This notebook shows how multiple retrieval strategies can be combined without additional LLM query decomposition.

---

## Tech Tools Used

- `Python 3.12+`
- `langchain` and related packages:
  - `langchain-core`
  - `langchain-openai`
  - `langchain-chroma`
  - `langchain-community`
  - `langchain-experimental`
  - `langchain-text-splitters`
- `OpenAI` for embeddings and LLM calls
- `ChromaDB` as a vector store
- `PyPDFLoader` for PDF ingestion
- `RecursiveCharacterTextSplitter` for document chunking
- `dotenv` for environment variable loading
- `ipykernel` for running notebooks

---

## Practical Takeaways

- RAG is stronger when retrieval is adapted to the query.
- Generating multiple sub-queries helps capture different angles of a question.
- Ensemble retrieval combines the strengths of multiple search strategies.
- Reciprocal Rank Fusion is a simple but effective fusion technique.
- Grounding the LLM answer with retrieved context is essential for accuracy.
- Prompt design should clearly instruct the model to use only the provided context.

---

## Running the Repo

1. Install dependencies:

```bash
cd c:\Generative-AI\RAG\8_Rag_fusion
python -m pip install -r requirements.txt
```

2. Create a `.env` file with your OpenAI API key:

```text
OPENAI_API_KEY=your_api_key_here
```

3. Open the notebooks with Jupyter:

```bash
jupyter notebook rag_pipeline.ipynb
jupyter notebook rag_pipeline_ensemble.ipynb
```

4. Run the notebook cells in order.

---

## Blueprint Summary

This repo is a compact, cognitive blueprint for RAG Fusion:

- Start with a document source (`PDF`)
- Split it into searchable chunks
- Convert text into embeddings and store them in a vector database
- Retrieve relevant chunks using one or more retrieval strategies
- Fuse the ranked results to produce a higher-quality context set
- Pass the fused context to an LLM prompt that is strictly grounded
- Generate an answer that is based on retrieved evidence

By following this flow, the repository shows how to reduce both retrieval errors and generation hallucinations.

---

## Interview Questions

1. What is Retrieval-Augmented Generation (RAG), and why is it useful?
2. Why do we split documents into chunks before embedding them?
3. What is the difference between similarity search and MMR retrieval?
4. How does Reciprocal Rank Fusion work, and why is it used?
5. What are the benefits of generating sub-queries from a main query?
6. Why do we need to ground the LLM answer with retrieved context?
7. What failure modes can occur if the LLM is allowed to answer without context?
8. How does an ensemble retriever differ from a single retriever?
9. What does `lambda_mult` control in MMR retrieval?
10. How would you measure the quality of a RAG system?
