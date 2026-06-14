# Advanced Retrievers: A Comprehensive Blueprint for RAG-Based Retrieval Strategies

<div align="center">

![Advanced Retrievers](https://img.shields.io/badge/Advanced%20RAG-Retrievers-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.12+-green?style=for-the-badge)
![LangChain](https://img.shields.io/badge/LangChain-Powered-orange?style=for-the-badge)

**A complete educational repository demonstrating advanced retrieval techniques for Retrieval-Augmented Generation (RAG) systems.**

</div>

---

## 📋 Table of Contents

1. [Repository Overview](#repository-overview)
2. [Core Intention](#core-intention)
3. [Tech Stack & Tools](#tech-stack--tools)
4. [Key Concepts & Takeaways](#key-concepts--takeaways)
5. [Complete Notebook Blueprint](#complete-notebook-blueprint)
6. [Deep Dive: Each Notebook Explained](#deep-dive-each-notebook-explained)
7. [How They Solve Real Problems](#how-they-solve-real-problems)
8. [Learning Path](#learning-path)
9. [Probable Interview Questions](#probable-interview-questions)
10. [Getting Started](#getting-started)

---

## 🎯 Repository Overview

This repository is a **complete blueprint** for understanding and implementing **advanced retrieval strategies** in Retrieval-Augmented Generation (RAG) systems. It demonstrates how to move beyond simple vector similarity search to build intelligent, context-aware retrieval systems that can understand user queries better, compress irrelevant information, and filter documents based on structured metadata.

### What Makes This Different?

Most RAG systems use **basic similarity search**: user query → vector embedding → cosine similarity → return top-K documents. This repository shows you the **advanced layer** that sits between the user and the vector database:

- **Query understanding** through LLM-powered query expansion and restructuring
- **Document compression** to extract only relevant portions, saving tokens
- **Metadata filtering** using natural language understanding
- **Hierarchical retrieval** that indexes small chunks but returns large context
- **Custom implementations** showing exactly how these techniques work under the hood

---

## 🎓 Core Intention

The intention of this repository is to **create RAG practitioners who understand**:

1. **Why** each retrieval technique exists (what problem it solves)
2. **How** each technique works (implementation details)
3. **When** to use each technique (trade-offs and scenarios)
4. **Custom implementations** of each technique (not just using out-of-the-box functions)

This repo answers: **"What happens between a user's question and the documents returned from the vector database?"**

---

## 🛠️ Tech Stack & Tools

### Core Dependencies

| Technology | Purpose | Version |
|------------|---------|---------|
| **LangChain** | Framework for building LLM applications | ^1.2.13 |
| **LangChain-OpenAI** | OpenAI API integration | ^1.1.12 |
| **LangChain-Chroma** | Chroma vector store integration | ^1.1.0 |
| **LangChain-Community** | Community retrievers (MultiQuery, ContextualCompression) | ^0.4.1 |
| **LangChain-Text-Splitters** | Document chunking utilities | ^1.1.1 |
| **Chroma** | Vector database (in-memory or persistent) | ^1.5.5 |
| **LangGraph** | Graph-based orchestration | ^1.1.3 |
| **Lark** | Parser for structured query language | ^1.3.1 |
| **PyPDF** | PDF document loading | ^6.9.2 |
| **Tiktoken** | OpenAI token counter | ^0.12.0 |
| **Python-dotenv** | Environment variable management | ^1.2.2 |

### Key Models Used

```
Embeddings: text-embedding-3-small
LLM: gpt-4-mini (cost-effective for demonstrations)
```

### Why These Technologies?

- **LangChain**: Provides abstractions for retrievers, text splitters, and composition patterns
- **Chroma**: Simple, fast vector database that doesn't require external servers
- **OpenAI Models**: State-of-the-art for understanding queries and extracting structured information
- **Lark**: Parses complex filter expressions (used in Self-Query retriever)

---

## 🧠 Key Concepts & Takeaways

### The Retrieval Pipeline

```
User Query
    ↓
[Query Understanding/Expansion]  ← Notebooks 3, 4, 7, 8
    ↓
[Similarity Search]
    ↓
[Document Compression/Filtering]  ← Notebooks 1, 2, 5, 6
    ↓
[Return Relevant Content]
```

### 8 Major Concepts Demonstrated

#### 1. **Contextual Compression** (Notebooks 1 & 2)
**What**: Filters retrieved documents to return only relevant portions
**Why**: LLMs charge per token; returning full documents wastes money and context window space
**How**: LLM reads each document and extracts only the relevant parts for the query
**Trade-off**: Slightly slower (extra LLM call) but saves significant token costs

#### 2. **Multi-Query Expansion** (Notebooks 3 & 4)
**What**: Rewrites a user's question into multiple phrasings
**Why**: Different phrasings might match different documents (recall problem)
**How**: LLM generates 3-5 alternative versions of the query; all are searched; results are deduplicated
**Trade-off**: ~3-5x more vector DB calls but catches documents that wouldn't match original query

#### 3. **Parent-Document Retrieval** (Notebooks 5 & 6)
**What**: Indexes small chunks but returns large parent documents
**Why**: Small chunks are precise for retrieval but lack context; large chunks have context but are hard to match
**How**: Create parent chunks (1500 tokens), split them into child chunks (400 tokens), index children, return parents
**Trade-off**: Two storage systems needed, more complex indexing

#### 4. **Self-Query Filtering** (Notebooks 7 & 8)
**What**: Uses LLM to parse natural language into structured metadata filters
**Why**: Simple keyword-based filters can't understand nuance; "movies from the 2000s" is hard to parse programmatically
**How**: LLM extracts both semantic query AND structured filters from natural language
**Trade-off**: One extra LLM call but enables semantic search + structured filtering simultaneously

### Cross-Cutting Insights

1. **The LLM as Retrieval Agent**: Modern RAG doesn't just embed text; it uses LLMs to understand what to search for
2. **Token Economy**: Every document returned, every query made costs tokens; optimization is crucial
3. **The Recall-Precision Tradeoff**: Multi-query improves recall (find more), compression improves precision (extract more relevant)
4. **Composition Over Monoliths**: These techniques combine (compression + multi-query, parent-doc + self-query)
5. **Scaling from In-Memory to Production**: Notebooks show both InMemoryStore and persistent LocalFileStore

---

## 📚 Complete Notebook Blueprint

### Notebook Structure at a Glance

```
Advanced_Retrievers/
├── notebooks/
│   ├── 1️⃣ contextual_compression.ipynb          ← LangChain built-in compression
│   ├── 2️⃣ custom_contextual_compression.ipynb    ← Your own compression implementation
│   ├── 3️⃣ multi_query.ipynb                      ← LangChain's multi-query expansion
│   ├── 4️⃣ custom_multi_query.ipynb               ← Your own multi-query implementation
│   ├── 5️⃣ parent_document_retriever.ipynb        ← LangChain's hierarchical retrieval
│   ├── 6️⃣ custom_parent_document_retriever.ipynb ← Your own parent-doc implementation
│   ├── 7️⃣ self_query.ipynb                       ← LangChain's LLM-powered filtering
│   └── 8️⃣ custom_self_query.ipynb                ← Your own self-query implementation
└── main.py                                        ← Entry point template
```

### The Build Progression

The notebooks follow a **"Built-in → Custom" pattern**:

- **Even notebooks (1, 3, 5, 7)**: Use LangChain's out-of-the-box implementations
- **Odd notebooks (2, 4, 6, 8)**: Rebuild the same functionality from scratch using LangChain's primitives

**Why this pattern?** You first see how it *should work*, then understand *how it actually works* by building it yourself.

---

## 🔍 Deep Dive: Each Notebook Explained

### Notebook 1 & 2: Contextual Compression

#### The Problem It Solves
```
Traditional Retrieval:
Query: "CRISPR personalized medicine"
↓
Returns entire 300+ token documents
Problem: Only 30 tokens are actually relevant
Wasted tokens → wasted money in LLM API calls
```

#### Notebook 1: Using LangChain's ContextualCompressionRetriever

```python
# Step 1: Create base retriever (standard similarity search)
vectorstore = InMemoryVectorStore.from_documents(docs, embeddings)
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Step 2: Add compression with LLMChainExtractor
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

# Result: Only relevant snippets returned, saving tokens!
```

**Three Compression Strategies Demonstrated**:

1. **LLMChainExtractor**: "Extract only the relevant parts" (uses LLM)
2. **EmbeddingsFilter**: Filter documents by embedding similarity threshold (no LLM call)
3. **DocumentCompressorPipeline**: Chain them (embeddings filter first → LLM extraction) to minimize LLM calls

**Key Takeaway**: Filter cheap first (embeddings), then use expensive operation (LLM) only on filtered results.

#### Notebook 2: Building Your Own Compression

```python
class CustomLLMExtractorChain:
    def compress_documents(self, documents, query):
        # Manually call LLM for each document
        for doc in documents:
            result = self.llm_chain.invoke({
                "question": query,
                "context": doc.page_content
            })
            # Only keep non-empty extractions
            if result and result != "NO_OUTPUT":
                yield Document(page_content=result, metadata=doc.metadata)

class CustomContextualCompressionRetriever(BaseRetriever):
    def _get_relevant_documents(self, query):
        # Base retrieval first
        docs = self.base_retriever.invoke(query)
        # Then compress
        return self.compressor.compress_documents(docs, query)
```

**Learning**: You see exactly where the LLM call happens and how metadata flows through the pipeline.

---

### Notebook 3 & 4: Multi-Query Expansion

#### The Problem It Solves
```
User's Question: "How are technologies improving health?"
Exact Match Problem: Vector DB might not match exact phrasing

Solution: Ask the same question 3 different ways:
1. "How do modern tech advances help medicine?"
2. "What technologies are revolutionizing healthcare?"
3. "Can you explain health tech innovations?"

Now more documents match!
```

#### Notebook 3: Using LangChain's MultiQueryRetriever

```python
# The magic is simple:
retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm,
    include_original=True  # Always include the original query
)

# Result: retriever.invoke() returns deduplicated union of all query variants
```

**Demonstrated Features**:
- Query variants are generated by LLM
- Results are deduplicated (no duplicates returned)
- `include_original=True` ensures original query is searched too
- Shows comparison: base retrieval (fewer docs) vs multi-query (more docs)

#### Notebook 4: Custom Implementation with Structured Output

```python
class QueriesSchema(BaseModel):
    queries: list[str] = Field(description="List of 3 alternative versions")

# LLM with structured output always returns a QueriesSchema
query_chain = prompt | llm.with_structured_output(QueriesSchema)

class CustomMultiQueryRetriever(BaseRetriever):
    def _get_relevant_documents(self, query):
        # Generate alternatives
        queries = self._generate_queries(query)  # Returns list[str]
        
        # Retrieve for each
        all_docs = []
        for q in queries:
            all_docs.extend(self.base_retriever.invoke(q))
        
        # Deduplicate by content
        return self._unique_documents(all_docs)
```

**Learning**: Structured output (Pydantic schemas) ensures the LLM output is always valid Python objects.

---

### Notebook 5 & 6: Parent-Document Retrieval

#### The Problem It Solves
```
The Context Window Dilemma:

Small Chunks (400 tokens):
✓ Easy to match semantically
✗ Lack surrounding context
✗ User gets orphaned snippets

Large Chunks (2000 tokens):
✓ Full context preserved
✗ Hard to match exactly
✗ Wastes context when mostly irrelevant

Solution: Index small chunks, return large parent chunks
```

#### Notebook 5: Using LangChain's ParentDocumentRetriever

```python
# Two splitters, different sizes
parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,      # Store these full documents
    chunk_overlap=200
)
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,       # Index these for searching
    chunk_overlap=50
)

# Two stores
store_memory = InMemoryStore()               # Holds parent chunks
vectorstore_memory = Chroma(...)            # Indexes child chunks

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore_memory,
    docstore=store_memory,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
    search_kwargs={'k': 3}
)

retriever.add_documents(docs)

# Key insight: child_count >> parent_count (many children per parent)
parent_count = 5        # 5 original documents
child_count = 45        # Split into 45 smaller searchable chunks
```

**Two Storage Options Demonstrated**:

1. **InMemoryStore**: Fast, no I/O, suitable for development
   ```python
   store = InMemoryStore()  # Python dict under the hood
   ```

2. **LocalFileStore**: Persistent on disk, for production
   ```python
   fs = LocalFileStore('./local_parent_store')
   store = create_kv_docstore(fs)  # Wraps with JSON serializer
   ```

#### Notebook 6: Custom Parent-Document Implementation

```python
class CustomParentDocumentRetriever(BaseRetriever):
    def add_documents(self, docs):
        # Step 1: Split into large parent chunks
        parent_chunks = self.parent_splitter.split_documents(docs)
        
        for parent in parent_chunks:
            # Step 2: Assign UUID to track parent
            parent_id = str(uuid.uuid4())
            
            # Step 3: Split parent into small child chunks
            children = self.child_splitter.split_documents([parent])
            
            # Step 4: Tag children with parent ID
            for child in children:
                child.metadata['parent_id'] = parent_id
            
            # Step 5: Index children for searching
            self.vectorstore.add_documents(children)
            
            # Step 6: Store parents by UUID
            self.docstore.mset([(parent_id, parent)])
    
    def _get_relevant_documents(self, query):
        # Search children
        child_docs = self.vectorstore.similarity_search(query, k=3)
        
        # Extract unique parent IDs from child metadata
        parent_ids = list(dict.fromkeys(
            doc.metadata['parent_id'] for doc in child_docs
        ))
        
        # Return parents
        return [doc for doc in self.docstore.mget(parent_ids)]
```

**Learning**: The UUID-based tracking pattern for maintaining hierarchical relationships.

---

### Notebook 7 & 8: Self-Query Retrieval

#### The Problem It Solves
```
Traditional Metadata Filtering:
search_kwargs = {
    "filter": {"$and": [
        {"genre": {"$eq": "sci-fi"}},
        {"year": {"$gte": 2000}}
    ]}
}

Problem: How does the user know the exact filter syntax?
User: "Give me sci-fi movies from the 2000s"
App: Somehow convert this to the filter above!

Solution: Use LLM to parse natural language into structured filters
```

#### Notebook 7: Using LangChain's SelfQueryRetriever

```python
# Tell the LLM what fields are filterable
metadata_field_info = [
    AttributeInfo(name="genre", description="Movie genre", type="string"),
    AttributeInfo(name="year", description="Release year", type="integer"),
    AttributeInfo(name="rating", description="IMDb rating", type="float"),
]

# LLM extracts both semantic query AND filters from user input
retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    document_contents="Brief plot descriptions",
    metadata_field_info=metadata_field_info,
    structured_query_translator=ChromaTranslator(),  # Translates to Chroma syntax
)

# Usage:
retriever.invoke("Sci-fi movies from after 2010")
# LLM internally:
# - Semantic query: "sci-fi movies"
# - Filters: {genre: "sci-fi", year: > 2010}
```

**The Magic**:
- LLM receives list of available fields (metadata_field_info)
- LLM decides which fields to filter on
- LLM generates both semantic query AND structured filters
- Translator converts to vector DB syntax (Chroma, Pinecone, etc.)

#### Notebook 8: Custom Self-Query Implementation

```python
class MetadataFilter(BaseModel):
    field: str
    value: str | int | float
    operator: str = "eq"  # eq, ne, gt, gte, lt, lte

class SelfQuerySchema(BaseModel):
    query: str                            # Semantic query
    filters: Optional[list[MetadataFilter]] = None

# LLM outputs a Pydantic object
class CustomSelfQueryRetriever(BaseRetriever):
    def _build_chroma_filter(self, filters):
        # Convert Pydantic to Chroma filter syntax
        op_map = {"eq": "$eq", "gt": "$gt", "gte": "$gte", ...}
        if len(filters) == 1:
            return {filters[0].field: {op_map[filters[0].operator]: filters[0].value}}
        return {"$and": [...]}  # Multiple filters
    
    def _get_relevant_documents(self, query):
        # Parse natural language into structured schema
        parsed = self.query_chain.invoke({"query": query})
        
        # Build filter for vector DB
        chroma_filter = self._build_chroma_filter(parsed.filters)
        
        # Search with both semantic query and structured filter
        return self.vectorstore.similarity_search(
            parsed.query,
            k=3,
            filter=chroma_filter
        )
```

**Key Insights**:
- Pydantic schemas enforce structure
- `with_structured_output()` binds schema to LLM
- Filter building is a simple dictionary transformation
- Vector DB gets both semantic search + metadata filter

---

## 🎯 How They Solve Real Problems

### Problem 1: Token Cost Explosion
```
Without Compression:
- 100 queries/day
- 3 results per query
- 300 tokens per result
- Total: 90,000 tokens/day just returning raw documents

With Compression:
- Same setup but documents compressed to 100 tokens
- Total: 30,000 tokens/day
- 67% reduction in API costs!
```

### Problem 2: Missing Relevant Documents
```
Query: "AI trends in healthcare"
Base Retrieval (3 results):
- Document about hospital AI systems
- Document about AI diagnosis tools
- Document about medical records

Missing: Documents about "machine learning in medicine" or "neural networks for health"

Multi-Query Expansion (9 deduplicated results):
Now includes all the above PLUS documents matching alternative phrasings
```

### Problem 3: Lack of Context
```
Small Chunk (400 tokens):
"CRISPR enables precise DNA modifications. Single-letter corrections possible."
→ User: "Where can I learn more?"

Large Parent Chunk (1500 tokens):
Full context about CRISPR, base editing, prime editing, clinical applications, 
regulatory landscape → User satisfied

Parent-Document Retrieval: Best of both worlds
```

### Problem 4: Structured + Semantic Search Disconnect
```
Traditional: Either semantic search OR structured filtering
User Query: "Show me action movies with >8.5 rating from 2010-2020"

Naive Approach:
- Search "action movies": returns all action films
- Filter rating > 8.5: reduces but misses nuance
- Filter year: too restrictive

Self-Query: Parses BOTH simultaneously
- Semantic understanding: quality action films from specific era
- Structured filters: exact year and rating ranges
- Result: Perfect match
```

---

## 📖 Learning Path

### For Beginners (Start Here)

1. **Day 1-2**: Read this README completely, understand the problems each technique solves
2. **Day 3**: Run Notebooks 1 (Compression) and 3 (Multi-Query) using LangChain built-ins
3. **Day 4-5**: Study Notebooks 2 and 4 to understand implementation details
4. **Day 6-7**: Run Notebooks 5 and 7 to see multi-store and filtering concepts

### For Intermediate Users

1. Modify notebooks to use your own documents (replace dummy docs)
2. Run the custom implementations (2, 4, 6, 8) to solidify understanding
3. Experiment with different chunk sizes in parent-document retriever
4. Try combining techniques (e.g., parent-doc + multi-query)

### For Advanced Users

1. Implement your own retrieval strategy combining multiple techniques
2. Benchmark the performance trade-offs (cost vs. recall vs. latency)
3. Use LangGraph to orchestrate complex retrieval pipelines
4. Add observability with LangSmith to trace actual performance

### Implementation Checklist

- [ ] Install all dependencies (`pip install -r requirements.txt`)
- [ ] Set up `.env` with `OPENAI_API_KEY`
- [ ] Run all 8 notebooks in sequence
- [ ] For each "custom" notebook, understand the implementation line-by-line
- [ ] Create a new notebook combining 2+ techniques
- [ ] Benchmark with real documents from your domain

---

## 📝 Probable Interview Questions

### Conceptual Questions

**Q1: "Explain the difference between contextual compression and multi-query retrieval."**

A: Contextual compression addresses the *token efficiency problem*—it keeps the same retrieved documents but returns only the relevant portions, saving tokens in the LLM context window. Multi-query retrieval addresses the *recall problem*—it retrieves more documents by searching with multiple query phrasings. Compression operates on retrieved documents; multi-query operates on the search queries.

**Q2: "When would you use parent-document retrieval instead of simple chunking?"**

A: Use parent-document retrieval when you need a tradeoff between search precision and context preservation. Simple chunking either gives small precise chunks (low context) or large context-rich chunks (hard to match). Parent-document retrieval indexes small chunks (precise matching) but returns large parent chunks (full context). This is especially valuable when each chunk needs surrounding context to be understandable—e.g., code files, legal documents, or scientific papers.

**Q3: "How does self-query retrieval differ from traditional keyword filtering?"**

A: Keyword filtering uses exact string matches or simple operators (AND, OR, NOT). Self-query uses an LLM to understand natural language and determine: (1) what the semantic search query should be, (2) what metadata filters to apply, and (3) how to combine them. This allows nuanced queries like "recent sci-fi movies with high ratings" to be parsed into both semantic understanding and precise filters. An LLM understands context; keyword systems don't.

**Q4: "Explain the token cost implications of multi-query retrieval."**

A: Multi-query tripled vector DB calls (search 3 variations) but on the LLM side, you generate only 1 LLM call (to create variants, if not using zero-shot). The benefit: you retrieve more relevant documents with better recall. The cost: more vector DB operations (cheaper than LLM) and slightly slower. It's a good tradeoff when coverage matters more than speed—e.g., in research or Q&A systems where missing documents is worse than being 200ms slower.

**Q5: "What's the advantage of custom implementations over using LangChain's built-ins?"**

A: Understanding implementations reveals exactly how the system works, revealing the assumptions and limitations. For example, the custom multi-query implementation shows you can easily change the deduplication strategy, the number of variants, or the format of generated queries. The custom self-query implementation shows how filters are translated to different vector DB syntaxes. This knowledge lets you adapt these techniques to your specific constraints and requirements.

### Implementation Questions

**Q6: "How would you combine contextual compression and multi-query retrieval?"**

A:
```python
# Multi-query expands recall
multi_query_retriever = MultiQueryRetriever.from_llm(...)

# Then compress for efficiency
compressor = LLMChainExtractor.from_llm(llm)
combined_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=multi_query_retriever  # Chain them!
)

# Result: More documents retrieved but compressed to essentials
```

**Q7: "In the parent-document retriever, why use UUID instead of simple indexing?"**

A: UUIDs are globally unique and don't depend on insertion order. This allows you to safely update, delete, or re-index documents without breaking parent-child relationships. Simple indices (0, 1, 2...) would break if you deleted a parent—all subsequent indices shift. UUIDs solve this. Additionally, UUIDs can be included in metadata, making the relationship explicit and traceable.

**Q8: "How would you handle documents from multiple sources with different metadata structures?"**

A:
```python
# Use optional fields in AttributeInfo
metadata_field_info = [
    AttributeInfo(name="title", description="...", type="string"),
    AttributeInfo(name="journal", description="...", type="string"),  # Optional
    AttributeInfo(name="company", description="...", type="string"),  # Optional
]

# Self-Query LLM filters only on present fields
# e.g., for academic papers → filters on "journal"
# for corporate docs → filters on "company"
```

**Q9: "What happens if the LLM generates invalid metadata filters in self-query retrieval?"**

A: With structured output (Pydantic), invalid data is caught at parse time—the LLM re-tries or returns an error. Without structured output, you'd need manual validation. The schema acts as a guard rail:

```python
class MetadataFilter(BaseModel):
    field: str
    operator: str = Field(pattern="^(eq|ne|gt|gte|lt|lte)$")  # Enum validation
    value: Union[str, int, float]
```

**Q10: "How would you measure the performance of different retrieval strategies?"**

A:
```
Metrics:
1. Recall: "Did we retrieve the truly relevant documents?" (F1 score vs. gold set)
2. Precision: "Were the retrieved documents actually relevant?" (Relevance scoring)
3. Cost: Total API calls × price
4. Latency: End-to-end retrieval time
5. Token efficiency: Tokens returned per query

Trade-off analysis:
- Base retrieval: Fast, cheap, but may miss documents (low recall)
- Multi-query: Slower, more expensive, better recall
- Compression: Same recall, fewer tokens, faster LLM inference
- Parent-doc: Slower indexing, better results for context-sensitive tasks

Choose based on your priorities (speed? coverage? cost?)
```

### System Design Questions

**Q11: "Design a RAG system for customer support that must minimize API costs."**

A:
```
Strategy:
1. First: Self-Query retriever to eliminate non-matching documents early
   (User: "Returns policy for electronics" → filter category=electronics)

2. Then: Parent-Document retriever for context
   (Index small policy snippets, return full policies)

3. Finally: Contextual compression to extract only relevant sections
   (E.g., warranty details from 5-page policy document)

Result: Fewer documents returned, heavily compressed, significant token savings
```

**Q12: "How would you handle very long documents (20+ pages) in RAG?"**

A:
```
Option 1: Parent-Document Retriever
- Parent = full section (e.g., "Warranty" section = 3 pages)
- Child = subsection (e.g., "Electronics warranty" = 1 page)
- Search on subsection level, return section level

Option 2: Multi-level hierarchy
- Index: sentences (precise matching)
- Return: paragraphs (some context)
- Provide: full sections (rich context) separately if needed

Option 3: Combination
- Use self-query to filter by document section/type first
- Then parent-document retrieval within that section
- Then compression if document still long
```

**Q13: "What are the failure modes of these retrieval techniques?"**

A:
```
Contextual Compression:
- Failure: LLM extracts too much → no token savings
- Failure: LLM extracts too little → loses important context
- Solution: Use embedding filter first to reduce LLM load

Multi-Query:
- Failure: Generated queries are too similar to original → no improvement
- Failure: Generated queries are off-topic → noise in results
- Solution: Use temperature > 0 for diversity, validate generations

Parent-Document:
- Failure: Parent is too large → loses precision
- Failure: Parent is too small → lacks context
- Solution: Tune chunk sizes based on document domain

Self-Query:
- Failure: LLM misunderstands which fields are filterable
- Failure: LLM generates impossible filters
- Solution: Provide clear examples in system prompt, use structured output
```

### Advanced Questions

**Q14: "How would you adapt self-query retrieval for a domain with 100+ metadata fields?"**

A:
```
Problem: LLM can't handle 100+ fields, attention exhaustion
Solution: Hierarchical filtering

1. Coarse filters: LLM decides "Document type" ∈ {paper, code, blog, dataset}
2. Fine filters: Route to type-specific handler
   - For papers: Filter by journal, year, citations
   - For code: Filter by language, framework, stars
3. Semantic search: Execute semantic query within filtered subset

This reduces field explosion and improves accuracy
```

**Q15: "Design a multi-stage retrieval pipeline optimizing for both latency and cost."**

A:
```
Pipeline:
Stage 1 (10ms, cheap): Self-Query filter
  - Reduce document space early
  - No LLM call yet (rules-based)

Stage 2 (50ms, medium): Multi-Query expansion on filtered space
  - Expand recall within pre-filtered subset
  - 1 LLM call for query generation

Stage 3 (30ms, fast): Parent-Document retrieval
  - Return larger chunks with context
  - Prepared during indexing

Stage 4 (100ms, expensive): Contextual compression on top results
  - Only compress top 3 documents
  - LLM call on smallest possible input

Total: ~190ms, cost minimized through staged filtering
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.12+
- OpenAI API key (set in `.env`)
- ~15 minutes to read this README thoroughly

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd Advanced_Retrievers

# Install dependencies
pip install -r requirements.txt

# Create .env file with your OpenAI API key
echo "OPENAI_API_KEY=your-key-here" > .env
```

### Running the Notebooks

```bash
# Using Jupyter
jupyter notebook notebooks/

# Or using VS Code
# Open any notebook and select kernel
```

### Recommended Execution Order

1. Read Notebook 1 (contextual_compression.ipynb) → understand the problem
2. Run Notebook 1 → see LangChain solution
3. Read Notebook 2 (custom_contextual_compression.ipynb) → see implementation
4. Run Notebook 2 → understand how it works
5. Repeat for Notebooks 3-4, 5-6, 7-8

### What to Modify to Learn

Each notebook has this structure:

```python
# Part 1: Setup (same in all notebooks)
from dotenv import load_dotenv
load_dotenv()
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4-mini", temperature=0)

# Part 2: Sample data
docs = [...]

# Part 3: Core technique (MODIFY THIS)
retriever = SomeRetriever(...)

# Part 4: Testing and comparison
results = retriever.invoke(query)
```

**Experiments to try**:
1. Change chunk sizes and observe retrieval differences
2. Use different embedding models
3. Use your own documents instead of sample data
4. Combine techniques
5. Add observability with LangSmith

---

## 📊 Comparison Table: When to Use Each Technique

| Technique | Problem Solved | Use When | Cost | Speed | Complexity |
|-----------|---|---|---|---|---|
| **Compression** | Token waste | Documents are large, API costs high | ↓ Token cost | +LLM latency | Low |
| **Multi-Query** | Low recall | Coverage matters, precision acceptable | ~3x vector DB | +Query time | Medium |
| **Parent-Doc** | Context loss | Need surrounding context, precise matching needed | Moderate | Depends on chunk size | Medium |
| **Self-Query** | Keyword limitations | Rich metadata, semantic + structured filters needed | +1 LLM call | +Parse time | Medium-High |
| **Combination** | Multiple issues | Production RAG systems with all constraints | Moderate | Optimized | High |

---

## 🔗 How to Think About RAG

```
Traditional Information Retrieval:
Query → Exact keyword match → Documents

Vector-Based RAG:
Query → Embedding → Semantic similarity → Documents

Advanced RAG (This Repository):
Query → [Understanding] → [Expansion/Filtering] → [Search] → [Compression] → Documents

The difference: Intelligence at every step, not just at the search step
```

---

## 🎓 Key Takeaways

1. **RAG is a pipeline, not a single operation**: Each stage can be optimized independently
2. **LLMs are search tools too**: Not just for generation, but for understanding intent
3. **Trade-offs exist everywhere**: Speed vs. cost, recall vs. precision, complexity vs. benefit
4. **Composition is powerful**: Simple techniques combined are better than complex single techniques
5. **Measure everything**: Without metrics, you don't know if your optimization actually helps
6. **Start simple, optimize later**: Base retrieval works; optimize when you identify bottlenecks

---

## 📚 Additional Resources

- [LangChain Documentation](https://python.langchain.com/)
- [Chroma Vector Database](https://www.trychroma.com/)
- [OpenAI API Reference](https://platform.openai.com/docs/)
- [RAG Best Practices](https://docs.llamaindex.ai/en/stable/)

---

## 🤝 Contributing & Questions

To deepen your understanding:
1. Fork the repository
2. Modify notebooks with your own data
3. Create custom implementations for other retrieval strategies
4. Document what you learn

---

<div align="center">

**Happy Learning! Master the fundamentals, and you'll build RAG systems that truly understand.**

*This repository transforms you from "I know what RAG is" to "I can implement advanced RAG correctly".*

</div>
