# RAG — From First Principles to a Production App

A hands-on journey through Retrieval-Augmented Generation: fifteen self-contained modules that build up
every component of a RAG system from scratch, and one complete application that puts them together.

Everything here runs. Every folder is independent — pick any one and start.

> **Credit where it's due.** The fifteen numbered modules follow the RAG curriculum from the
> [CampusX](https://learnwith.campusx.in/) course, which is where I learned this. The topic sequence and
> the teaching examples are theirs; the implementations here are my own work from following along, and
> none of the course's notes, slides, or written material is reproduced. If you find this useful, the
> course is worth your money.
>
> [**Papeer**](./rag-papeer-project-main) started from the course project and went a long way past it —
> cross-encoder reranking, inline citations with source pruning, conversation-aware query resolution,
> per-session isolation with full cleanup, a rebuilt UI, and Docker deployment. The
> [changelog](./rag-papeer-project-main/README.md#changelog) records what changed and why.

---

## Two ways to use this repo

**📚 Learning the components** → work through the numbered folders `1_` to `15_`. Each is a standalone
project with its own notebooks, dependencies, and README. They're ordered, but you can jump straight to
whatever you need.

**🚀 Seeing it all assembled** → go to [`rag-papeer-project-main/`](./rag-papeer-project-main) — a
deployed research-paper assistant built on top of these ideas.

---

## Start here

| If you are… | Start with |
|---|---|
| **New to RAG** | `1_Document_Loaders` → `2_Text_Splitters` → `3_Embedding_Models` → `4_Vector_Stores` → `5_Retrievers`. That's the full pipeline: load, chunk, embed, store, retrieve. |
| **Comfortable with basic RAG** | `6_Advanced_Retrievers` → `7_Rerankers` → `8_Rag_fusion` → `9_Rag_Hyde`. These are the techniques that separate a demo from something that works. |
| **Building agentic systems** | `10_Corrective_Rag` → `11_Agentic_Rag` → `12_Graph_Rag` → `13_Multimodal_Rag`. |
| **Shipping to production** | `14_RAGAS` (how do you know it works?) → `15_Rag_guardrails` (what stops it misbehaving?) → the Papeer project. |
| **Here for the project** | Jump straight to [`rag-papeer-project-main/`](./rag-papeer-project-main). |

---

## Papeer — the full application

A research-paper assistant that answers questions about your own papers and shows its work. Upload a PDF,
paste a URL, or pull a paper from ArXiv, then ask.

- **Answers with receipts** — every claim carries a citation pointing to a specific paper and page, with a
  side panel showing the exact passages used
- **Checks whether findings still hold** — searches recent literature and surfaces superseding papers
- **Routes intelligently** — decides per question whether it needs your documents, a live web search, or
  neither
- **Follows a conversation** — *"what about its limitations?"* resolves against what you were just discussing

Built with LangGraph, Qdrant, FlashRank, and Streamlit; containerised with Docker. The
[project README](./rag-papeer-project-main/README.md) has architecture diagrams, design-decision notes, and
setup instructions.

---

## The modules

### Foundations — the core pipeline

| Module | What it covers |
|---|---|
| **[1_Document_Loaders](./1_Document_Loaders)** | Getting text out of things: PDF, CSV, JSON, plain text, and web pages |
| **[2_Text_Splitters](./2_Text_Splitters)** | Chunking strategies — character, recursive, document-aware, semantic, and LLM-based |
| **[3_Embedding_Models](./3_Embedding_Models)** | Turning text into vectors, with OpenAI (hosted) and Ollama (local) |
| **[4_Vector_Stores](./4_Vector_Stores)** | ChromaDB: CRUD operations, persisting a database, and a full PDF pipeline |
| **[5_Retrievers](./5_Retrievers)** | Similarity search, score thresholds, MMR, BM25, hybrid search, and a custom ensemble retriever |

### Retrieval quality — making it actually work

| Module | What it covers |
|---|---|
| **[6_Advanced_Retrievers](./6_Advanced_Retrievers)** | Contextual compression, multi-query, parent-document, and self-query — each implemented *twice*, once with the library and once from scratch |
| **[7_Rerankers](./7_Rerankers)** | Cross-encoder reranking with Cohere (hosted) and FlashRank (local, free) |
| **[8_Rag_fusion](./8_Rag_fusion)** | Generating query variations and fusing the ranked results |
| **[9_Rag_Hyde](./9_Rag_Hyde)** | HyDE — embed a *hypothetical answer* instead of the question, so short queries match dense prose |

### Architectures — beyond a single retrieval pass

| Module | What it covers |
|---|---|
| **[10_Corrective_Rag](./10_Corrective_Rag)** | Six notebooks building CRAG step by step: basic RAG → retrieval refinement → an evaluator → web-search fallback → query rewriting → handling ambiguity |
| **[11_Agentic_Rag](./11_Agentic_Rag)** | Letting the model decide: conditional retrieval, tool use, retrieval evaluation with rewriting, and query decomposition |
| **[12_Graph_Rag](./12_Graph_Rag)** | Knowledge-graph RAG — ingestion and retrieval as separate notebooks |
| **[13_Multimodal_Rag](./13_Multimodal_Rag)** | Two competing strategies: converting images to text, versus true multimodal embeddings |

### Production concerns

| Module | What it covers |
|---|---|
| **[14_RAGAS](./14_RAGAS%28RAG_Evaluation%29)** | Measuring a RAG system: faithfulness, context precision, context recall, response relevancy, noise sensitivity |
| **[15_Rag_guardrails](./15_Rag_guardrails)** | PII detection, toxicity, jailbreak attempts, topic restriction, competitor mentions, and response validation |

---

## Running any module

Each folder is an independent project with its own `pyproject.toml` and lockfile. Nothing is shared, so
there's no global environment to set up and no dependency conflicts between modules.

Using [`uv`](https://docs.astral.sh/uv/) (recommended):

```bash
cd 5_Retrievers
uv sync
uv run jupyter lab      # then open the notebooks/ folder
```

Or with plain pip:

```bash
cd 5_Retrievers
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

### API keys

Most modules need an OpenAI key. Create a `.env` file inside the module folder:

```env
OPENAI_API_KEY=sk-...
```

A few modules need extras — Cohere for `7_Rerankers`, Tavily and Qdrant for the Papeer project. Each
module's own README lists exactly what it requires. **Nothing here needs a paid tier to explore**, though
embedding a large PDF will consume credits.

Want to run everything locally with no API cost? `3_Embedding_Models` covers Ollama embeddings, and
`7_Rerankers` covers FlashRank, which runs on CPU.

---

## Notes

- Modules are numbered for reading order, not dependency — each one stands alone.
- `Advanced_Retrievers` (unnumbered) is a leftover from a rename; use `6_Advanced_Retrievers`.
- Notebooks are committed with their outputs, so you can read the results without running anything.

---

*Questions, corrections, and pull requests are welcome. If something here is unclear or wrong, open an issue —
that's genuinely useful feedback.*
