import os
import re
import sqlite3
import warnings
from typing import Annotated

warnings.filterwarnings("ignore", message="The default value of `allowed_objects`")

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import InjectedToolCallId, tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import InjectedState, ToolNode, tools_condition
from langgraph.types import Command
from pydantic import BaseModel, Field
from tavily import TavilyClient

from backend.models import ClaimVerificationResult, RelevancyDecision, RouterDecision
from backend.reranker import candidate_pool_size, rerank
from backend.vector_store import search as vs_search

load_dotenv()

llm = ChatOpenAI(model="gpt-5.4-mini")


# ── State ─────────────────────────────────────────────────────────────────────

class RAGState(MessagesState):
    session_id: str
    query: str
    route: str | None
    retrieved_docs: list[Document]
    retrieval_attempts: int
    claim_verdict: str | None
    claim_source: str | None
    superseding_papers: list[dict] | None
    answer: str | None
    user_question: str | None
    sources: list[dict] | None
    is_relevant: bool | None
    rewrite_count: int


# ── Conversation history ──────────────────────────────────────────────────

# Appended beneath cited answers. Defined here so history extraction and
# rendering can never drift apart.
SOURCES_MARKER = "\n\n---\n\n**Sources**\n\n"

# Marks queries this graph generated itself (rewrites), so they are never
# replayed as though the user had typed them.
SYNTHETIC = "papeer_synthetic"

MAX_HISTORY_MESSAGES = 10


def _strip_sources(text: str) -> str:
    return text.split(SOURCES_MARKER)[0] if isinstance(text, str) else text


def conversation_history(
    messages: list,
    limit: int | None = MAX_HISTORY_MESSAGES,
    strip_sources: bool = True,
) -> list[dict]:
    """The durable user/assistant transcript, with agent scratchpad removed.

    `state["messages"]` interleaves three different things: real conversation,
    tool traffic (tool-call AIMessages and their ToolMessages), and synthetic
    rewrite queries. Only the first belongs in a prompt — replaying tool traffic
    wastes tokens and replaying rewrites makes the model think the user asked
    something they never asked.
    """
    turns: list[dict] = []
    for msg in messages or []:
        kind = type(msg).__name__
        if kind == "HumanMessage":
            if (getattr(msg, "additional_kwargs", None) or {}).get(SYNTHETIC):
                continue
            turns.append({"role": "user", "content": msg.content})
        elif kind in ("AIMessage", "AIMessageChunk"):
            if getattr(msg, "tool_calls", None):
                continue  # scratchpad: the model asking for a tool, not talking
            if isinstance(msg.content, str) and msg.content.strip():
                content = _strip_sources(msg.content) if strip_sources else msg.content
                turns.append({"role": "assistant", "content": content})
    return turns[-limit:] if limit else turns


def prior_turns(messages: list) -> list[dict]:
    """History with the current question removed — it is passed separately."""
    turns = conversation_history(messages)
    while turns and turns[-1]["role"] == "user":
        turns.pop()
    return turns


# ── Contextualization ──────────────────────────────────────────────────────

CONTEXTUALIZE_SYSTEM = (
    "Rewrite the user's latest message into a standalone question that makes sense "
    "without the conversation.\n\n"
    "- Resolve pronouns and references ('it', 'that paper', 'his work') against the conversation.\n"
    "- Expand bare confirmations ('yes please', 'go ahead', 'sure', 'tell me more') into the "
    "specific request the assistant just offered. If the assistant offered several options and "
    "the user agreed without choosing, fold them into one question.\n"
    "- If the message is already standalone, return it completely unchanged.\n"
    "- Never answer the question. Return only the rewritten question, with no preamble."
)


def contextualize_query_node(state: RAGState) -> dict:
    """Resolve follow-ups before anything downstream reads the query.

    Without this, the router classifies 'Yes, please.' on its own and retrieval
    searches for the literal string 'Yes, please.' — both meaningless.
    """
    messages = state["messages"]
    latest = messages[-1].content if messages else state.get("query", "")
    history = prior_turns(messages)

    # First turn of a session: nothing to resolve, so skip the LLM call.
    if not history:
        return {"query": latest, "user_question": latest}

    transcript = "\n".join(f"{t['role']}: {t['content'][:800]}" for t in history)
    try:
        response = llm.invoke([
            {"role": "system", "content": CONTEXTUALIZE_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Conversation so far:\n{transcript}\n\n"
                    f"Latest message: {latest}\n\nStandalone question:"
                ),
            },
        ])
        resolved = (response.content or "").strip() or latest
    except Exception:
        resolved = latest  # never block a turn on the rewrite

    return {"query": resolved, "user_question": resolved}


# ── Router ────────────────────────────────────────────────────────────────────

ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a routing assistant for a research paper Q&A system. "
        "Classify the user query into exactly one of three categories:\n\n"
        "  retrieve — Use this for TWO types of questions:\n"
        "    (a) Questions about the content of uploaded research papers "
        "(e.g. methods, results, conclusions, authors).\n"
        "    (b) Questions that require live or current information that cannot be "
        "answered from general knowledge alone — such as current events, today's weather, "
        "live prices, recent news, or anything where the answer changes over time "
        "(e.g. 'Who is the current president?', 'What is the price of gold today?', "
        "'What is the weather in Delhi?').\n"
        "  verify_claim — The user wants to check whether a specific claim or finding "
        "from a paper is still accurate or has been superseded.\n"
        "  direct_answer — A stable general knowledge question answerable from training data "
        "with no retrieval needed (e.g. 'What is softmax?', 'Who invented the transformer?', "
        "'Explain backpropagation.').\n\n"
        "When in doubt between retrieve and direct_answer, prefer retrieve.\n\n"
        "Return only the route field.",
    ),
    ("human", "{query}"),
])

router_chain = ROUTER_PROMPT | llm.with_structured_output(RouterDecision)


def router_node(state: RAGState) -> dict:
    # Already resolved to a standalone question by contextualize_query_node.
    query = state.get("query") or state["messages"][-1].content
    decision: RouterDecision = router_chain.invoke({"query": query})
    return {"route": decision.route}


# ── Tool schemas ──────────────────────────────────────────────────────────────

class RetrieverInput(BaseModel):
    query: str = Field(description="Semantic query to search research paper chunks")
    k: int = Field(default=4, ge=1, le=10, description="Number of chunks to retrieve")


class WebSearchInput(BaseModel):
    optimized_query: str = Field(description="Query rewritten and optimized for web search")
    max_results: int = Field(default=3, ge=1, le=10, description="Number of web results to return")


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool(args_schema=RetrieverInput)
def retrieve_from_vectorstore(
    query: str,
    k: int,
    session_id: Annotated[str, InjectedState("session_id")],
    current_docs: Annotated[list, InjectedState("retrieved_docs")],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> list:
    """Search the uploaded research paper vector store for relevant passages."""
    # Overfetch cheaply with the bi-encoder, then let the cross-encoder pick the
    # best `k`. See backend/reranker.py for why.
    candidates = vs_search(query=query, session_id=session_id, k=candidate_pool_size(k))
    if not candidates:
        return [ToolMessage(content="No relevant documents found in the vector store.", tool_call_id=tool_call_id)]
    docs = rerank(query, candidates, top_n=k)
    summary = f"Retrieved {len(docs)} chunk(s), reranked from {len(candidates)} candidate(s)."
    return [
        ToolMessage(content=summary, tool_call_id=tool_call_id),
        Command(update={"retrieved_docs": (current_docs or []) + docs}),
    ]


@tool(args_schema=WebSearchInput)
def web_search(
    optimized_query: str,
    max_results: int,
    current_docs: Annotated[list, InjectedState("retrieved_docs")],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> list:
    """Search the web for current or supplementary information using Tavily."""
    client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    results = client.search(optimized_query, max_results=max_results)
    if not results.get("results"):
        return [ToolMessage(content="No web results found.", tool_call_id=tool_call_id)]
    web_docs = [
        Document(
            page_content=r["content"],
            metadata={"url": r["url"], "title": r.get("title", "Web Result")},
        )
        for r in results["results"]
    ]
    summary = f"Found {len(web_docs)} web result(s) for: {optimized_query}"
    return [
        ToolMessage(content=summary, tool_call_id=tool_call_id),
        Command(update={"retrieved_docs": (current_docs or []) + web_docs}),
    ]


# ── Retrieval agent singletons ────────────────────────────────────────────────

RETRIEVAL_TOOLS = [retrieve_from_vectorstore, web_search]
retrieval_llm = llm.bind_tools(RETRIEVAL_TOOLS, parallel_tool_calls=False)
base_tool_node = ToolNode(RETRIEVAL_TOOLS)

RETRIEVE_SYSTEM = (
    "You are a research assistant gathering context to answer a user's question about research papers.\n\n"
    "You have two tools available and full control over how you use them:\n\n"
    "1. retrieve_from_vectorstore — searches the uploaded paper collection.\n"
    "   You decide:\n"
    "   - query: the semantic search query (phrase it to best match relevant paper chunks)\n"
    "   - k: how many chunks to retrieve (1–10; use more for broad questions, fewer for specific ones)\n\n"
    "2. web_search — searches the live web via Tavily.\n"
    "   You decide:\n"
    "   - optimized_query: rewrite the user's question as a concise, keyword-rich web search query\n"
    "   - max_results: how many results to fetch (1–10)\n\n"
    "Choose the right source based on the question:\n"
    "- Questions about the uploaded papers → use retrieve_from_vectorstore\n"
    "- Questions about current events, recent developments, or supplementary information → use web_search\n"
    "- Call only one tool per turn.\n\n"
    "Do NOT produce a final answer. Only call tools to collect context."
)


# ── Relevancy check ───────────────────────────────────────────────────────────

RELEVANCY_CHECK_SYSTEM = (
    "You are evaluating whether retrieved document chunks are relevant enough "
    "to answer a user's question about research papers.\n\n"
    "Return is_relevant=true if the chunks contain information that meaningfully "
    "addresses the question — even partially. "
    "Return is_relevant=false only if the chunks are clearly off-topic or contain "
    "no useful information.\n\nBe lenient: if there is any substantive overlap, return true."
)

relevancy_llm = llm.with_structured_output(RelevancyDecision)

QUERY_REWRITE_SYSTEM = (
    "You are a query rewriting assistant for a research paper retrieval system. "
    "The previous query failed to retrieve relevant document chunks. "
    "Rewrite the query using more specific or alternative terminology, "
    "domain-specific keywords, or a narrower sub-question.\n\n"
    "Return ONLY the rewritten query as plain text. No explanation, no preamble."
)


# ── Nodes ─────────────────────────────────────────────────────────────────────

def agent_node(state: RAGState) -> dict:
    current_attempts = state.get("retrieval_attempts", 0)
    # Once at the cap, use plain LLM so the agent cannot emit more tool calls.
    # This prevents orphaned tool_call IDs from entering the persisted message history.
    # retrieval llm --> tool call --> tool result
    # llm --> no tools are bounded --> tool call
    lm = llm if current_attempts >= MAX_RETRIEVAL_ATTEMPTS else retrieval_llm
    system = RETRIEVE_SYSTEM
    resolved = state.get("query")
    latest = state["messages"][-1].content if state["messages"] else None
    if resolved and resolved != latest:
        # The raw last message may be an elliptical follow-up ("yes please").
        # Hand the agent the resolved question so its search queries are usable.
        system += (
            "\n\nThe user's latest message resolves, in context, to this standalone "
            f"question:\n{resolved}\n"
            "Base your search queries on it rather than on the literal last message."
        )
    messages = [{"role": "system", "content": system}] + state["messages"]
    response = lm.invoke(messages)
    updates: dict = {"messages": [response]}
    if getattr(response, "tool_calls", None):
        updates["retrieval_attempts"] = current_attempts + 1
    return updates


def relevancy_check_node(state: RAGState) -> dict:
    query = state["query"]
    docs = state.get("retrieved_docs") or []
    doc_snippets = "\n\n---\n\n".join(doc.page_content[:300] for doc in docs[:3])
    if not doc_snippets:
        return {"is_relevant": False}
    prompt = (
        f"Question: {query}\n\nRetrieved chunks:\n{doc_snippets}\n\n"
        "Are these chunks relevant to answering the question?"
    )
    decision: RelevancyDecision = relevancy_llm.invoke([
        {"role": "system", "content": RELEVANCY_CHECK_SYSTEM},
        {"role": "user", "content": prompt},
    ])
    return {"is_relevant": decision.is_relevant}


def query_rewrite_node(state: RAGState) -> dict:
    original_query = state["query"]
    rewrite_count = state.get("rewrite_count", 0)
    response = llm.invoke([
        {"role": "system", "content": QUERY_REWRITE_SYSTEM},
        {"role": "user", "content": f"Original query: {original_query}\n\nWrite an improved search query."},
    ])
    rewritten = response.content.strip()
    return {
        # Tagged so it is never replayed to the model as a real user turn.
        "messages": [
            HumanMessage(content=rewritten, additional_kwargs={SYNTHETIC: True})
        ],
        "query": rewritten,
        "retrieved_docs": [],
        "retrieval_attempts": 0,
        "rewrite_count": rewrite_count + 1,
        "is_relevant": None,
    }


CLAIM_ANALYSIS_PROMPT = (
    "You are a research fact-checker. Given a claim from a research paper and "
    "a set of recent web and arXiv search results, determine:\n"
    "1. Has this claim been superseded, significantly challenged, or updated by more recent work?\n"
    "2. Identify up to 3 papers from the provided results that supersede or update the claim.\n\n"
    "Rules:\n"
    "- Use ONLY titles and URLs that appear verbatim in the provided search results.\n"
    "- Prefer arXiv paper links (arxiv.org) over general web links when available.\n"
    "- For each superseding paper, write one sentence explaining how it supersedes the claim.\n"
    "- If the claim still holds, set is_superseded=false and return an empty superseding_papers list.\n"
    "- verdict_summary should be 1-2 sentences suitable for display to the user."
)

verification_llm = llm.with_structured_output(ClaimVerificationResult)


def verify_claim_node(state: RAGState) -> dict:
    claim = state.get("query") or state["messages"][-1].content
    tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

    # General web search for recent work superseding the claim
    general_results = tavily_client.search(
        f"recent research superseding: {claim[:200]}",
        max_results=5,
    ).get("results", [])

    # arXiv-targeted search via web to get paper titles and links
    arxiv_results = tavily_client.search(
        f"site:arxiv.org {claim[:200]}",
        max_results=5,
    ).get("results", [])

    # Build context block
    lines = ["=== General Web Search Results ==="]
    for r in general_results:
        lines.append(
            f"Title: {r.get('title', '')}\n"
            f"URL: {r['url']}\n"
            f"Snippet: {r.get('content', '')[:300]}\n"
        )

    lines.append("=== arXiv Paper Search Results ===")
    for r in arxiv_results:
        lines.append(
            f"Title: {r.get('title', '')}\n"
            f"URL: {r['url']}\n"
            f"Snippet: {r.get('content', '')[:300]}\n"
        )

    context = "\n".join(lines)

    prompt = (
        f"{CLAIM_ANALYSIS_PROMPT}\n\n"
        f"Claim to verify:\n{claim}\n\n"
        f"Search Results:\n{context}"
    )
    result: ClaimVerificationResult = verification_llm.invoke([
        {"role": "user", "content": prompt}
    ])

    papers_dicts = [p.model_dump() for p in result.superseding_papers[:3]]
    return {
        "claim_verdict": result.verdict_summary,
        "claim_source": papers_dicts[0]["url"] if papers_dicts else None,
        "superseding_papers": papers_dicts,
    }


# ── Citations ─────────────────────────────────────────────────────────────────

def _source_key(doc: Document) -> tuple:
    """Chunks from the same page (or the same web page) share one citation number."""
    md = doc.metadata or {}
    url = md.get("url") or md.get("source") or ""
    if isinstance(url, str) and url.startswith(("http://", "https://")):
        return ("web", url)
    title = md.get("title") or "Untitled document"
    return ("doc", title, md.get("page"))


def _source_label(key: tuple, md: dict) -> tuple[str, str | None]:
    """Return (human-readable label, url or None) for one citation entry."""
    if key[0] == "web":
        return (md.get("title") or key[1]), key[1]
    title, page = key[1], key[2]
    if isinstance(page, int):
        return f"{title} — p. {page + 1}", None
    return title, None


def build_cited_context(docs: list[Document]) -> tuple[str, list[dict]]:
    """Build a numbered context block plus the matching ordered source list."""
    order: list[tuple] = []
    grouped: dict[tuple, dict] = {}
    for doc in docs:
        key = _source_key(doc)
        if key not in grouped:
            grouped[key] = {"metadata": doc.metadata or {}, "chunks": []}
            order.append(key)
        grouped[key]["chunks"].append(doc.page_content)

    sources: list[dict] = []
    blocks: list[str] = []
    for n, key in enumerate(order, start=1):
        label, url = _source_label(key, grouped[key]["metadata"])
        sources.append({"n": n, "label": label, "url": url})
        header = f"[{n}] {label}" + (f" ({url})" if url else "")
        blocks.append(header + "\n" + "\n\n".join(grouped[key]["chunks"]))

    return "\n\n---\n\n".join(blocks), sources


_CITATION_RE = re.compile(r"\[(\d+)\]")


def prune_citations(answer: str, sources: list[dict]) -> tuple[str, list[dict]]:
    """Drop sources the model never cited, then renumber the rest from 1.

    Listing retrieved-but-unused sources makes an answer look less grounded than
    it is, and invites the reader to check a page that says nothing relevant.
    Renumbering happens in a single regex pass so that a swap (3->1, 1->2) can't
    double-map.
    """
    valid = {s["n"] for s in sources}
    cited: list[int] = []
    for match in _CITATION_RE.finditer(answer):
        n = int(match.group(1))
        if n in valid and n not in cited:
            cited.append(n)

    # Model ignored the citation instruction — keep every source rather than
    # silently stripping attribution from the answer.
    if not cited:
        return answer, sources

    remap = {old: new for new, old in enumerate(cited, start=1)}
    answer = _CITATION_RE.sub(
        lambda m: f"[{remap[int(m.group(1))]}]" if int(m.group(1)) in remap else m.group(0),
        answer,
    )
    by_n = {s["n"]: s for s in sources}
    pruned = [
        {**by_n[old], "n": new}
        for old, new in sorted(remap.items(), key=lambda kv: kv[1])
    ]
    return answer, pruned


def render_sources(sources: list[dict]) -> str:
    """Markdown block appended beneath the generated answer."""
    if not sources:
        return ""
    entries = [
        f"{s['n']}. [{s['label']}]({s['url']})" if s["url"] else f"{s['n']}. {s['label']}"
        for s in sources
    ]
    return SOURCES_MARKER + "\n".join(entries)


ANSWER_SYSTEM = (
    "You are a research assistant answering questions about academic papers.\n\n"
    "Citation rules:\n"
    "- Support every factual claim with a bracketed citation, e.g. "
    "'The model uses multi-head attention [1].'\n"
    "- Place the citation immediately after the claim it supports, not at the end "
    "of the paragraph.\n"
    "- Cite several sources at once when a claim draws on more than one: [1][3].\n"
    "- Use ONLY the source numbers listed below. Never invent a number.\n"
    "- Do NOT write your own Sources or References list — one is appended automatically.\n"
    "- If the sources do not answer the question, say so plainly instead of guessing."
)


DIRECT_ANSWER_SYSTEM = (
    "You are a research assistant. Answer from your own knowledge.\n"
    "Use the conversation above to resolve references and to honour anything you "
    "previously offered to do — if the user is accepting an earlier offer, carry it out "
    "rather than asking them to restate the question."
)


def generate_answer_node(state: RAGState) -> dict:
    route = state.get("route")
    # `query` may have been overwritten by a retrieval rewrite; answer what the
    # user actually asked.
    query = state.get("user_question") or state["query"]
    history = prior_turns(state["messages"])
    sources: list[dict] = []

    if route == "retrieve":
        if state.get("is_relevant") is False and state.get("rewrite_count", 0) >= 1:
            answer = (
                "I wasn't able to find relevant information in the uploaded papers "
                "to answer your question. You may want to rephrase your question "
                "or upload additional papers."
            )
        else:
            docs = state.get("retrieved_docs") or []
            if not docs:
                answer = "I don't know the answer."
            else:
                context, sources = build_cited_context(docs)
                prompt = (
                    f"Numbered sources:\n\n{context}\n\n"
                    f"Question: {query}\n\n"
                    "Answer the question using the sources above, citing each claim."
                )
                answer = llm.invoke(
                    [
                        {"role": "system", "content": ANSWER_SYSTEM},
                        *history,
                        {"role": "user", "content": prompt},
                    ]
                ).content
                answer, sources = prune_citations(answer, sources)
                answer += render_sources(sources)

    elif route == "verify_claim":
        verdict = state.get("claim_verdict", "")
        papers = state.get("superseding_papers") or []
        claim_text = query
        if papers:
            papers_block = "\n\n".join(
                f"{i + 1}. **{p['title']}**\n   {p['summary']}\n   Link: {p['url']}"
                for i, p in enumerate(papers)
            )
            answer = (
                f"**Claim Verification Result**\n\n"
                f"> {claim_text}\n\n"
                f"**Verdict:** {verdict}\n\n"
                f"**Superseding Papers:**\n\n{papers_block}\n\n"
                f"---\n"
                f"*You can load any of these papers into your knowledge base "
                f"to continue your research with the latest findings.*"
            )
        else:
            answer = (
                f"**Claim Verification Result**\n\n"
                f"> {claim_text}\n\n"
                f"**Verdict:** {verdict}\n\n"
                f"*No papers directly superseding this claim were found in recent literature.*"
            )

    else:  # direct_answer
        answer = llm.invoke(
            [
                {"role": "system", "content": DIRECT_ANSWER_SYSTEM},
                *history,
                {"role": "user", "content": query},
            ]
        ).content

    return {"answer": answer, "sources": sources, "messages": [AIMessage(content=answer)]}


# ── Graph ─────────────────────────────────────────────────────────────────────

MAX_RETRIEVAL_ATTEMPTS = 3


def route_query(state: RAGState) -> str:
    return state["route"]


def agent_routing(state: RAGState) -> str:
    # Always execute pending tool calls first — shortcutting here would leave
    # an AIMessage with tool_calls unmatched by ToolMessages in the checkpointer,
    # corrupting history for all future turns in the same session.
    tc = tools_condition(state)
    if tc == "tools":
        return "retrieval"
    if state.get("retrieval_attempts", 0) >= MAX_RETRIEVAL_ATTEMPTS:
        return "generate_answer"
    return "relevancy_check"


def after_relevancy_routing(state: RAGState) -> str:
    if state.get("is_relevant", False):
        return "generate_answer"
    if state.get("rewrite_count", 0) < 1:
        return "query_rewrite"
    return "generate_answer"


def build_graph(db_path: str = "checkpoints.db"):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    graph = StateGraph(RAGState)
    graph.add_node("contextualize", contextualize_query_node)
    graph.add_node("router", router_node)
    graph.add_node("agent_node", agent_node)
    graph.add_node("retrieval", base_tool_node)
    graph.add_node("relevancy_check", relevancy_check_node)
    graph.add_node("query_rewrite", query_rewrite_node)
    graph.add_node("verify_claim", verify_claim_node)
    graph.add_node("generate_answer", generate_answer_node)

    graph.set_entry_point("contextualize")
    graph.add_edge("contextualize", "router")

    graph.add_conditional_edges(
        "router",
        route_query,
        {
            "retrieve": "agent_node",
            "verify_claim": "verify_claim",
            "direct_answer": "generate_answer",
        },
    )

    graph.add_conditional_edges(
        "agent_node",
        agent_routing,
        {
            "retrieval": "retrieval",
            "relevancy_check": "relevancy_check",
            "generate_answer": "generate_answer",
        },
    )
    graph.add_edge("retrieval", "agent_node")

    graph.add_conditional_edges(
        "relevancy_check",
        after_relevancy_routing,
        {"query_rewrite": "query_rewrite", "generate_answer": "generate_answer"},
    )
    graph.add_edge("query_rewrite", "agent_node")

    graph.add_edge("verify_claim", "generate_answer")
    graph.add_edge("generate_answer", END)

    return graph.compile(checkpointer=checkpointer)

