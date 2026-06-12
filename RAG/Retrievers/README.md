# RAG Retrievers: A Complete Guide to Document Retrieval in Retrieval-Augmented Generation

## 📋 Table of Contents
1. [Repository Overview](#repository-overview)
2. [What is a Retriever?](#what-is-a-retriever)
3. [Retriever Types & Concepts](#retriever-types--concepts)
4. [Project Structure](#project-structure)
5. [Technology Stack](#technology-stack)
6. [Detailed Notebook Guide](#detailed-notebook-guide)
7. [Key Takeaways](#key-takeaways)
8. [Interview Questions](#interview-questions)

---

## 🎯 Repository Overview

This repository is a **complete educational blueprint** for understanding and implementing document retrievers in Retrieval-Augmented Generation (RAG) systems. It demonstrates six progressive retriever strategies—from basic similarity search to advanced hybrid ensemble approaches—with hands-on Jupyter notebooks and working code examples.

### 🎓 Who Should Use This?
- **Beginners in RAG/NLP**: Learn retriever fundamentals from scratch
- **ML Engineers**: Understand different retrieval strategies and their trade-offs
- **Developers**: Implement production-ready retriever systems
- **Researchers**: Compare semantic vs. keyword-based approaches

### 💡 Central Goal
To provide a **crystal-clear roadmap** showing **why** different retriever strategies exist, **how** they work, and **when** to use each one. This repo proves that effective RAG depends as much on smart document retrieval as it does on language models.

---

## 🧠 What is a Retriever?

### The Problem
When an LLM needs to answer a question, it only "knows" what's in its training data. If you want it to answer questions about your proprietary documents, you need to:
1. Store your documents somewhere searchable
2. Find the most relevant documents for the user's question
3. Feed those documents to the LLM as context

A **retriever** is the component that solves step 2—finding relevant documents quickly and accurately.

### The Basic Flow
```
User Query → [RETRIEVER] → Retrieved Documents → LLM → Answer
```

### Why It Matters
- **Quality of retrieval directly impacts answer quality** — If the retriever misses relevant documents, the LLM can't use them
- **Speed matters** — Searching through millions of documents must be fast
- **Different queries need different strategies** — Keyword searches work differently than semantic searches

---

## 🔍 Retriever Types & Concepts

### 1. **Semantic (Dense) Retrievers**
**Concept**: Convert queries and documents into dense vectors (embeddings), then measure similarity.

**How it works**:
- Documents and queries are embedded into fixed-size vectors (e.g., 1536 dimensions)
- Similarity is measured using cosine similarity, L2 distance, or inner product
- Returns documents semantically closest to the query

**Pros**:
- ✅ Understands meaning, not just keywords
- ✅ Works across synonyms ("vehicle" matches "car")
- ✅ Works with vague or conceptual queries

**Cons**:
- ❌ Requires embedding model (computational cost)
- ❌ May miss exact keyword matches
- ❌ Embedding quality depends on training data

**Example Query**: *"How do cells generate energy?"* → Finds documents about mitochondria, ATP, respiration even if they don't contain "energy"

---

### 2. **Keyword-Based (Sparse) Retrievers (BM25)**
**Concept**: Match query terms against document terms using TF-IDF variants.

**How it works**:
- Builds an inverted index of terms → documents
- Scores documents based on term frequency and inverse document frequency
- No embeddings needed—pure token matching

**Pros**:
- ✅ Fast and lightweight (no embedding model)
- ✅ Deterministic—same query always returns same results
- ✅ Excellent for exact keyword matches
- ✅ Works well for domain-specific terminology

**Cons**:
- ❌ No semantic understanding
- ❌ Fails on paraphrased queries
- ❌ Struggles with short documents

**Example Query**: *"antibiotic bacterial infection"* → Finds documents containing these exact or similar terms

---

### 3. **Similarity Score Threshold**
**Concept**: Filter semantic search results by a minimum similarity score.

**Why it matters**:
- Prevents low-quality results from being returned
- Trade-off: strict threshold may return no results, loose threshold may include noise
- Useful when you need **precision over recall**

**Use case**: Medical information retrieval where low-confidence matches are risky

---

### 4. **Maximal Marginal Relevance (MMR)**
**Concept**: Balance relevance (semantic similarity) with diversity (dissimilarity between results).

**The Problem It Solves**: 
Imagine searching for "deep learning training" and getting three near-identical documents about gradient descent. They're all relevant but redundant. MMR penalizes similar results and picks diverse ones.

**How it works**:
- Fetches top-k candidate documents by similarity
- Iteratively selects documents that are:
  - **Relevant** to the query (high similarity)
  - **Diverse** from already-selected documents (low similarity to each other)
- Controlled by `lambda_mult` parameter:
  - `lambda_mult=1.0`: Pure relevance (like similarity search)
  - `lambda_mult=0.5`: Balanced relevance + diversity
  - `lambda_mult=0.0`: Pure diversity

**Result**: Returned documents cover different aspects of the topic instead of repeating the same information.

---

### 5. **Hybrid Search (Semantic + Keyword)**
**Concept**: Combine dense and sparse retrievers to get the best of both worlds.

**Why it works**:
- **Dense search** finds semantically related documents (even with paraphrased queries)
- **Sparse search** finds exact keyword matches (even if semantically unrelated)
- Together they cover more ground

**Example**:
Query: *"How do vaccines work?"*
- **BM25 alone**: Returns documents containing "vaccine" (docs 1-2)
- **Semantic alone**: Returns documents about immune system (docs 3-5)
- **Hybrid**: Returns both sets, providing comprehensive coverage

**Merging Strategy**: Reciprocal Rank Fusion (RRF)
- Combines rankings from multiple retrievers
- Formula: `score(doc) = sum(weight_i / (rank_i + k))`
- Weights allow prioritizing one retriever over another
- `k` (typically 60) smooths the contribution of lower-ranked results

---

### 6. **Custom Ensemble Retriever**
**Concept**: Implement your own retriever logic by extending `BaseRetriever`.

**Why it matters**:
- You can implement custom scoring logic
- Combine retrievers in custom ways
- Add business logic (e.g., document recency, source reliability)
- Educational value—understand exactly how fusion works

**Key Implementation Details**:
- Inherit from `langchain_core.retrievers.BaseRetriever`
- Implement `_get_relevant_documents()` method
- Manually implement RRF or other fusion algorithms
- Fully customizable and transparent

---

## 📁 Project Structure

```
Retrievers/
├── main.py                          # Entry point (placeholder)
├── pyproject.toml                   # Project metadata and dependencies
├── requirements.txt                 # Python package dependencies
├── README.md                        # This file
└── notebooks/
    ├── 01_similarity_search.ipynb              # Baseline: semantic search
    ├── 02_similarity_score_threshold.ipynb     # Add confidence filtering
    ├── 03_mmr.ipynb                           # Diversity-aware ranking
    ├── 04_bm25.ipynb                          # Keyword-based retrieval
    ├── 05_hybrid_search.ipynb                 # Combine semantic + keyword
    └── 06_custom_ensemble_retriever.ipynb     # Custom RRF implementation
```

---

## 🛠 Technology Stack

### Core Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| **LangChain** | ≥1.2.12 | Retriever interfaces, composition, and abstractions |
| **LangChain-Community** | ≥0.4.1 | BM25Retriever and other community retrievers |
| **LangChain-OpenAI** | ≥1.1.11 | OpenAI embedding model integration |
| **LangChain-Chroma** | ≥1.1.0 | ChromaDB vector store integration |
| **ChromaDB** | ≥1.5.5 | Vector database for storing and searching embeddings |
| **rank-bm25** | ≥0.2.2 | BM25 algorithm implementation |
| **python-dotenv** | ≥1.2.2 | Load API keys from .env files |
| **ipykernel** | ≥7.2.0 | Jupyter kernel for running notebooks |

### Embedding Model
- **text-embedding-3-small** (OpenAI): 1536-dimensional embeddings, fast and efficient

### Vector Store
- **ChromaDB**: In-memory vector database with HNSW indexing, cosine similarity support

### Environment
- **Python**: 3.12+
- **Jupyter**: For interactive notebook execution

---

## 📚 Detailed Notebook Guide

### **Notebook 1: Similarity Search (01_similarity_search.ipynb)**

**What You'll Learn**:
- How to create vector embeddings using OpenAI
- Store documents in a vector store (ChromaDB)
- Perform semantic similarity search
- Use retrievers in LangChain chains

**Key Concepts**:
- **Vector embeddings**: Dense representations capturing document meaning
- **Cosine similarity**: Dot product divided by vector magnitudes—ranges from -1 to 1
- **Retriever interface**: LangChain abstraction for composable retrieval

**Code Walkthrough**:
1. Load environment variables (OpenAI API key)
2. Create 12 sample documents across topics: space, biology, history, geography
3. Initialize OpenAI embeddings model
4. Store documents in ChromaDB vector store
5. Create a retriever with `search_type="similarity"` and `k=2`
6. Execute queries and retrieve top-2 most similar documents

**Expected Results**:
```
Query: "How do rockets work?"
Result 1: "Rockets work by expelling gas..."  [topic: space]
Result 2: "Spacecraft use gravitational slingshots..."  [topic: space]
```

**Why This Matters**: Foundation for all semantic search—understanding how meaning is captured and searched.

---

### **Notebook 2: Similarity Score Threshold (02_similarity_score_threshold.ipynb)**

**What You'll Learn**:
- Filter results by confidence threshold
- Understand similarity score distribution
- Adjust precision vs. recall trade-off

**Key Concepts**:
- **Similarity score**: Distance between query and document embeddings (0-1 range)
- **Precision**: Of results returned, how many are relevant
- **Recall**: Of all relevant documents, how many did we find
- **Threshold tuning**: Higher threshold = higher precision, lower recall

**Code Walkthrough**:
1. Create 12 documents: 5 ML docs (highly relevant), 3 economics, 2 cooking, 2 sports (off-topic)
2. Query: "How does ML model training work?"
3. Retrieve top-3 results with similarity scores
4. Convert scores to actual cosine similarity: `actual_similarity = 1 - score`
5. Apply threshold filter (e.g., threshold=0.43)
6. Only documents exceeding threshold are returned

**Expected Results**:
```
Threshold=0.43 → Returns 3 ML documents
Threshold=0.50 → Returns 1-2 top ML documents
Threshold=0.60 → Likely returns 0 documents (too strict)
```

**Practical Use**: Use in production when you need high confidence answers—better to return nothing than wrong information.

---

### **Notebook 3: Maximal Marginal Relevance (03_mmr.ipynb)**

**What You'll Learn**:
- Problem with redundant results from similarity search
- How MMR balances relevance and diversity
- Tuning the relevance-diversity trade-off

**Key Concepts**:
- **Redundancy problem**: Semantic search returns similar documents
- **Diversity**: Covering different aspects of a topic
- **lambda_mult**: Parameter controlling trade-off (0 = pure diversity, 1 = pure relevance)
- **fetch_k vs. k**: Fetch more candidates before re-ranking

**Code Walkthrough**:
1. Create documents with intentional redundancy:
   - 3 near-identical gradient descent docs (deep learning)
   - 3 diverse deep learning concepts (dropout, batch norm, learning rate scheduler)
   - Off-topic docs (climate, art, law)
2. Query: "deep learning model training and optimization"
3. Compare similarity search vs. MMR with different lambda values:
   - lambda_mult=1.0: Returns 3 redundant gradient descent docs
   - lambda_mult=0.5: Returns gradient descent + dropout + batch norm (diverse!)
   - lambda_mult=0.0: Returns most dissimilar docs (loses relevance)

**Expected Results**:
```
λ=1.0 (relevance): [GD doc 1, GD doc 2, GD doc 3] — Redundant
λ=0.5 (balanced): [GD doc 1, Dropout, Batch Norm] — Diverse yet relevant
λ=0.0 (diversity): [Random diverse docs] — No relevance
```

**When to Use**: When you want coverage of multiple aspects (e.g., "Tell me about COVID-19" should cover vaccines, symptoms, variants, etc.)

---

### **Notebook 4: BM25 Keyword-Based Retrieval (04_bm25.ipynb)**

**What You'll Learn**:
- Keyword-based retrieval without embeddings
- When semantic search fails vs. when keyword search excels
- TF-IDF and BM25 algorithms

**Key Concepts**:
- **BM25 (Best Matching 25)**: Probabilistic ranking function based on term frequency
- **TF (Term Frequency)**: How often a term appears in a document
- **IDF (Inverse Document Frequency)**: How rare the term is across all documents
- **Sparse retrieval**: No dense vectors—only term indices

**Code Walkthrough**:
1. Create 12 documents: medicine, architecture, finance, literature
2. Create BM25 retriever from raw documents (no embeddings)
3. Test three queries:
   - **Query 1 (keyword match)**: "antibiotic bacterial infection treatment" 
     - BM25 ✅ Finds medicine docs with exact terms
   - **Query 2 (keyword match)**: "compound interest returns stakes in company"
     - BM25 ✅ Finds finance docs with matching terms
   - **Query 3 (semantic)**: "a structure that feels light and with windows"
     - BM25 ❌ Misses Gothic cathedral doc (no keyword overlap)

**Expected Results**:
```
Query 1 ✅ BM25: Returns antibiotic & vaccine docs
Query 2 ✅ BM25: Returns compound interest & stock docs
Query 3 ❌ BM25: Might return architecture docs but not the right one
```

**Key Insight**: BM25 is great for domains with specific terminology (medical, legal, technical) but struggles with natural language variations.

---

### **Notebook 5: Hybrid Search / Ensemble Retrieval (05_hybrid_search.ipynb)**

**What You'll Learn**:
- Combine semantic and keyword retrievers
- Use EnsembleRetriever with Reciprocal Rank Fusion
- Leverage strengths of both approaches
- Weight different retrievers

**Key Concepts**:
- **Reciprocal Rank Fusion (RRF)**: Combines rankings from multiple systems
  - RRF formula: `score = Σ(weight_i × 1/(rank_i + k))`
  - `k=60` smoothing constant prevents rank-1 bias
  - Weights allow prioritizing certain retrievers
- **Hybrid retrieval**: Semantic + keyword = comprehensive coverage

**Code Walkthrough**:
1. Create 12 documents: health (vaccine docs), programming, history, nature
2. Initialize two retrievers:
   - **Dense**: ChromaDB semantic search (k=4)
   - **Sparse**: BM25Plus (k=2, bm25_variant="plus")
3. Create EnsembleRetriever with weights [0.8, 0.2] (80% semantic, 20% keyword)
4. Query: "How do vaccines work to protect against diseases?"

**Retriever Comparison**:
```
BM25 Results:
  [1] "Vaccines work by introducing weakened pathogen..." (exact match)
  [2] "The flu vaccine is reformulated each year..." (exact match)

ChromaDB Results:
  [1] "The immune system produces antibodies..." (semantic match)
  [2] "Herd immunity occurs when..." (semantic match)
  [3] "B-lymphocytes produce proteins..." (semantic match)
  [4] "Vaccines work by introducing..." (semantic match)

Ensemble Results (RRF):
  [1] "Vaccines work..." (rank 1 from both)
  [2] "The immune system produces antibodies..." (from semantic)
  [3] "Herd immunity occurs..." (from semantic)
  [4] "B-lymphocytes produce proteins..." (from semantic)
```

**Practical Benefit**: Ensemble catches documents from both strategies—exact keyword matches AND semantically similar documents.

---

### **Notebook 6: Custom Ensemble Retriever Implementation (06_custom_ensemble_retriever.ipynb)**

**What You'll Learn**:
- Extend LangChain's BaseRetriever class
- Implement Reciprocal Rank Fusion algorithm manually
- Build production-ready custom retrievers
- Understand RRF scoring in detail

**Key Concepts**:
- **BaseRetriever**: Abstract class for all LangChain retrievers
- **_get_relevant_documents()**: Core method to implement
- **RRF scoring**: Manual implementation showing exact calculations
- **Document deduplication**: Handle same document from multiple retrievers

**Code Walkthrough**:

1. **Custom Retriever Class**: `MyEnsembleRetriever`
   ```python
   class MyEnsembleRetriever(BaseRetriever):
       retrievers: List[BaseRetriever]
       weights: List[float]
       rrf_k: int = 60
   ```

2. **Core Algorithm**:
   ```python
   doc_scores = {}
   for retriever_idx, results in enumerate(all_results):
       weight = weights[retriever_idx]
       for rank, doc in enumerate(results):
           # RRF formula: weight × 1/(rank + k)
           rrf_score = weight * (1.0 / (rank + rrf_k))
           # Accumulate scores for duplicate docs
           if doc in doc_scores:
               doc_scores[doc] += rrf_score
           else:
               doc_scores[doc] = rrf_score
   ```

3. **Return sorted results**:
   ```python
   sorted_docs = sorted(doc_scores.items(), 
                       key=lambda x: x[1], 
                       reverse=True)
   ```

**Why Implement It Yourself?**:
- ✅ Understand the algorithm deeply
- ✅ Add custom scoring logic
- ✅ Debug specific retrieval issues
- ✅ Implement domain-specific fusion strategies

**Results**: Same as notebook 5, but with full transparency of how scoring works.

---

## 🎓 Key Takeaways

### 1. **Retrieval Quality Directly Impacts LLM Answers**
The retriever is the foundation. A perfect LLM with poor retrieval = poor answers.

### 2. **No One-Size-Fits-All Solution**
- **Semantic search**: Paraphrased queries, conceptual matching
- **Keyword search**: Specific terminology, domain-specific docs
- **Hybrid**: Coverage of both patterns

### 3. **Trade-offs Are Central**
| Aspect | Semantic | Keyword | Hybrid |
|--------|----------|---------|--------|
| Speed | ⚠️ Slower | ✅ Fast | ⚠️ Moderate |
| Semantic understanding | ✅ Excellent | ❌ Poor | ✅ Good |
| Exact matches | ⚠️ Approximate | ✅ Perfect | ✅ Perfect |
| Embedding cost | ❌ Required | ✅ None | ⚠️ Required |
| Result diversity | ❌ Redundant | ⚠️ Limited | ✅ Diverse (with MMR) |

### 4. **Diversity Matters**
Different documents covering different aspects > Multiple redundant documents

### 5. **LangChain Abstractions Enable Composition**
Retrievers are composable building blocks—stack, weight, and combine them flexibly.

### 6. **Always Benchmark with Your Data**
Optimal retriever choice depends on:
- Document length and style
- Query type (keyword vs. conceptual)
- Domain (medical, legal, technical requires keywords; creative/conceptual requires semantic)
- Latency requirements

### 7. **Monitoring & Thresholds Are Production-Ready**
- Monitor retrieval accuracy (are retrieved docs actually relevant?)
- Use thresholds for high-stakes applications
- Adjust weights in ensemble based on performance

---

## 🧑‍💼 Probable Interview Questions

### **Beginner Level**

**Q1: What is a retriever and why do we need it in RAG?**
**A**: A retriever finds relevant documents from a knowledge base to provide context to an LLM. Without a good retriever, the LLM can only use what's in its training data—retrievers let it answer questions about new, proprietary, or real-time documents.

**Q2: What's the difference between semantic and keyword-based retrieval?**
**A**: 
- Semantic: Understands meaning, finds paraphrased matches, but slower and requires embeddings
- Keyword: Fast, deterministic, perfect for exact matches, but can't understand synonyms or concepts

**Q3: When would you use BM25 instead of embeddings?**
**A**: When you have domain-specific terminology where exact keywords matter (medical, legal), when latency is critical and you can't embed documents, or when you have very short documents where embeddings are unreliable.

---

### **Intermediate Level**

**Q4: Explain Maximal Marginal Relevance (MMR) and give an example.**
**A**: MMR balances relevance (similarity to query) with diversity (dissimilarity between results). Example: Searching "COVID-19" should return docs about vaccines AND symptoms AND variants—not three docs all about vaccines. The `lambda_mult` parameter tunes this trade-off.

**Q5: What's Reciprocal Rank Fusion and how does it work?**
**A**: RRF merges rankings from multiple retrievers by scoring each document based on its rank in each retriever: `score = Σ(weight_i / (rank_i + k))`. The `k` constant (usually 60) ensures lower-ranked results still contribute. It's order-aware—a document ranked #1 in one retriever gets credit even if completely absent from another.

**Q6: Why would you use a hybrid (semantic + keyword) retriever?**
**A**: Because they solve different problems:
- Semantic finds conceptually similar but differently-worded docs
- Keyword finds exact terminology matches
Together they maximize recall. You pay 2x retrieval cost but get more comprehensive results.

---

### **Advanced Level**

**Q7: How would you implement a custom retriever that boosts recent documents or from trusted sources?**
**A**: Extend `BaseRetriever` and implement `_get_relevant_documents()`. After getting results from base retrievers:
1. Apply semantic/BM25 ranking
2. Apply custom scoring: `score *= (recency_boost × source_credibility_score)`
3. Re-sort by adjusted scores
4. Return top-k

This lets you inject domain-specific logic into retrieval.

**Q8: Compare the three retrieval strategies at scale (millions of documents).**
**A**:
- **Semantic**: O(1) lookup via HNSW indexing, but embedding millions of docs is expensive upfront. Query latency: ~100ms
- **Keyword**: O(log n) inverted index lookup, extremely fast. Query latency: ~10ms
- **Hybrid**: Runs both, then merges—~110ms but comprehensive

For millions of docs, hybrid with optimized weights is often best.

**Q9: What happens if your embedding model and documents were trained on different distributions?**
**A**: Embedding quality suffers. Example: If embeddings trained on news but documents are medical, semantic similarity is meaningless. Solutions:
1. Fine-tune embeddings on your domain
2. Use keyword retrieval for domain-specific terms
3. Use hybrid retrieval to compensate
4. Validate with human judges

**Q10: How do you evaluate retrieval quality?**
**A**: Metrics include:
- **Recall@k**: Of truly relevant docs, what % appear in top-k results?
- **Precision@k**: Of returned top-k, what % are actually relevant?
- **MRR (Mean Reciprocal Rank)**: Average rank of first relevant document
- **NDCG (Normalized Discounted Cumulative Gain)**: Ranking quality considering position
- **Human evaluation**: Most reliable but expensive

---

### **Scenario-Based Questions**

**Q11: You're building a RAG system for a law firm. What retriever strategy would you use?**
**A**: **Primarily BM25 with semantic as backup**. Reasoning:
- Legal language is precise; keyword matches are critical ("statute of limitations" must match exactly)
- Domain-specific terminology matters more than semantic similarity
- Speed is important (lawyers work under time pressure)
- Hybrid as fallback: if BM25 returns few results, semantic search provides options

**Q12: A user complains that your RAG system returns three very similar documents instead of covering different angles on their question.**
**A**: **Use MMR instead of pure similarity search**. Set `lambda_mult=0.6` to balance relevance with diversity. This tells the system: "Give me relevant results, but avoid redundancy." The ranking will pick diverse documents.

**Q13: Your retriever works perfectly on small test data but fails on production data (1M documents). What could be wrong?**
**A**: Common issues:
1. **Embedding quality at scale**: Dense index (HNSW) may have insufficient node connections—adjust `efConstruction` parameter
2. **Memory/latency**: Retrieving semantically on 1M docs takes time—add caching or use hybrid (BM25 is fast)
3. **Data drift**: Test docs are clean; production docs are messier—add preprocessing
4. **Embedding model limitations**: "text-embedding-3-small" may struggle on your domain—fine-tune or use larger model

---

## 🚀 How to Use This Repository

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Create .env file with your OpenAI API key
echo "OPENAI_API_KEY=your_key_here" > .env
```

### Run Notebooks
```bash
# Start Jupyter
jupyter notebook

# Open notebooks in order: 01 → 06
# Each builds on concepts from previous notebooks
```

### Experiment
- Modify queries to see how retrievers respond
- Adjust parameters (`k`, `lambda_mult`, threshold, weights)
- Add your own documents and test
- Compare retriever outputs

---

## 📖 Further Reading

- [LangChain Retrievers Documentation](https://python.langchain.com/docs/modules/data_connection/retrievers/)
- [Reciprocal Rank Fusion Paper](https://dl.acm.org/doi/10.1145/1571941.1572114)
- [Maximal Marginal Relevance](https://arxiv.org/abs/2107.03810)
- [BM25 Algorithm](https://en.wikipedia.org/wiki/Okapi_BM25)
- [Vector Embeddings & Similarity](https://huggingface.co/course/en/chapter5/1)

---

## 📝 Summary Sheet (Quick Reference)

| Retriever | When to Use | Speed | Cost | Strength |
|-----------|------------|-------|------|----------|
| **Similarity Search** | General semantic matching | ⚠️ Moderate | ⚠️ Embedding cost | Semantic understanding |
| **Similarity + Threshold** | High-confidence retrieval | ⚠️ Moderate | ⚠️ Embedding cost | Precision-focused |
| **MMR** | Diverse results wanted | ⚠️ Moderate | ⚠️ Embedding cost | Coverage of topics |
| **BM25** | Exact keywords matter | ✅ Fast | ✅ Free | Speed + determinism |
| **Hybrid** | Comprehensive coverage | ⚠️ Slower | ⚠️ 2x cost | Best of both |
| **Custom Ensemble** | Domain-specific logic | ⚠️ Custom | ⚠️ Custom | Flexibility |

---

**Created**: 2026
**Purpose**: Educational blueprint for understanding document retrievers in RAG systems  
**Audience**: Beginners to advanced practitioners in AI/ML  

*This README is a complete reference guide. Start with notebook 01, progress sequentially, and refer back to this guide for concepts.*
