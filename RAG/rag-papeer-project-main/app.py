import json
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from backend.btw_handler import handle_btw
from backend.paper_loader import load_arxiv, load_document, load_webpage
from backend.rag_graph import build_graph, conversation_history, delete_session_thread
from backend.vector_store import add_paper, delete_session as drop_session_vectors, list_papers

st.set_page_config(page_title="Papeer", page_icon="📚", layout="wide")


def _inject_css() -> None:
    """Load the visual layer. Absent or broken CSS just falls back to stock Streamlit."""
    css_path = Path("assets/style.css")
    if css_path.exists():
        st.markdown(
            f"<style>{css_path.read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True,
        )


_inject_css()


@st.cache_resource
def get_graph():
    return build_graph()


SESSIONS_FILE = Path("sessions.json")
_rename_llm = ChatOpenAI(model="gpt-5-mini")


def load_sessions() -> dict:
    try:
        return json.loads(SESSIONS_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_sessions(sessions_meta: dict) -> None:
    SESSIONS_FILE.write_text(json.dumps(sessions_meta, indent=2), encoding="utf-8")


def _serialize_state(values: dict) -> dict:
    out = {}
    for k, v in values.items():
        if k == "messages":
            out[k] = [
                {
                    "type": type(m).__name__,
                    "content": (
                        m.content[:300]
                        if isinstance(m.content, str)
                        else repr(m.content)[:300]
                    ),
                }
                for m in (v or [])
            ]
        elif k == "retrieved_docs":
            out[k] = [
                {"content": d.page_content[:300], "metadata": d.metadata}
                for d in (v or [])
            ]
        else:
            out[k] = v
    return out


# ── Live progress labels for the graph run ────────────────────────────────────

NODE_STAGES = {
    "contextualize": "🧵 Reading the conversation so far…",
    "router": "🧭 Working out how to answer…",
    "agent_node": "🤔 Deciding what to look up…",
    "retrieval": "📚 Fetching passages…",
    "relevancy_check": "🔎 Checking the passages actually answer your question…",
    "query_rewrite": "♻️ Nothing useful found — refining the search query…",
    "verify_claim": "🕵️ Checking the claim against recent literature…",
    "generate_answer": "✍️ Writing the answer…",
}

TOOL_STAGES = {
    "retrieve_from_vectorstore": "📚 Searching your papers…",
    "web_search": "🌐 Searching the web…",
}

# Shown under each answer so the reader knows why it looks the way it does.
ROUTE_BADGES = {
    "retrieve": "Answered from your documents",
    "verify_claim": "Claim checked against recent literature",
    "direct_answer": "Answered from general knowledge — no retrieval",
}


def _tool_stage(node_update) -> str | None:
    """If a node emitted tool calls, name the tool that is about to run."""
    if not isinstance(node_update, dict):
        return None
    for msg in node_update.get("messages") or []:
        for call in getattr(msg, "tool_calls", None) or []:
            label = TOOL_STAGES.get(call.get("name"))
            if label:
                return label
    return None


def generate_session_name(first_message: str) -> str:
    try:
        response = _rename_llm.invoke(
            [
                {
                    "role": "system",
                    "content": (
                        "Generate a concise 3-5 word title for a research chat session "
                        "based on the user's first message. Return only the title, "
                        "no punctuation at the end, no quotes."
                    ),
                },
                {"role": "user", "content": first_message[:500]},
            ]
        )
        return response.content.strip()
    except Exception:
        return "New Session"


def maybe_rename_session(session_id: str, first_message: str) -> None:
    if st.session_state.sessions_meta.get(session_id, {}).get("is_named"):
        return
    name = generate_session_name(first_message)
    st.session_state.sessions_meta[session_id]["name"] = name
    st.session_state.sessions_meta[session_id]["is_named"] = True
    save_sessions(st.session_state.sessions_meta)


def create_session() -> str:
    sid = str(uuid.uuid4())
    st.session_state.sessions_meta[sid] = {
        "id": sid,
        "name": "New Session",
        "created_at": datetime.now().isoformat(),
        "is_named": False,
    }
    save_sessions(st.session_state.sessions_meta)
    st.session_state.chats[sid] = []
    st.session_state.turns[sid] = 0
    return sid


def load_session_chats(session_id: str) -> list[dict]:
    config = {"configurable": {"thread_id": session_id}}
    try:
        state = graph.get_state(config)
        if not state or not state.values:
            return []
        chats = []
        turn = 0
        # Reuse the graph's transcript filter so replay matches what the model
        # sees: no tool traffic, no synthetic rewrite queries, no blank bubbles.
        for entry in conversation_history(
            state.values.get("messages", []), limit=None, strip_sources=False
        ):
            if entry["role"] == "user":
                chats.append({"role": "user", "content": entry["content"]})
            else:
                turn += 1
                chats.append(
                    {
                        "role": "assistant",
                        "content": entry["content"],
                        "turn": turn,
                        "graph_state": {},
                        "route": None,
                    }
                )
        return chats
    except Exception:
        return []


def switch_session(session_id: str) -> None:
    st.session_state.active_session_id = session_id
    if session_id not in st.session_state.chats:
        st.session_state.chats[session_id] = load_session_chats(session_id)
    if session_id not in st.session_state.turns:
        turn_count = sum(
            1 for m in st.session_state.chats[session_id] if m["role"] == "assistant"
        )
        st.session_state.turns[session_id] = turn_count


def delete_session(session_id: str) -> None:
    """Remove a session everywhere it exists.

    Session data lives in three places: the metadata JSON, a Qdrant collection,
    and a checkpointer thread. Dropping only the first would orphan the other
    two — abandoned collections in particular accumulate against the Qdrant
    free-tier quota with nothing left in the UI pointing at them.

    Vector deletion is attempted first: it is the one that costs something if
    skipped, and if it fails the session stays visible so it can be retried.
    """
    try:
        drop_session_vectors(session_id)
    except Exception as e:
        st.error(f"Could not delete stored documents — session kept. ({e})")
        return

    try:
        delete_session_thread(graph, session_id)
    except Exception:
        pass  # orphaned rows are harmless; the session is gone from the UI

    st.session_state.sessions_meta.pop(session_id, None)
    save_sessions(st.session_state.sessions_meta)
    st.session_state.chats.pop(session_id, None)
    st.session_state.turns.pop(session_id, None)
    st.session_state.pop(f"processed_files_{session_id}", None)
    st.session_state.pending_delete = None

    if st.session_state.get("active_session_id") == session_id:
        remaining = sorted(
            st.session_state.sessions_meta.values(),
            key=lambda s: s["created_at"],
            reverse=True,
        )
        if remaining:
            switch_session(remaining[0]["id"])
        else:
            st.session_state.active_session_id = create_session()


def latest_retrieved_docs(session_id: str) -> list[dict]:
    """Passages behind the most recent answer, for the context pane.

    Read from the checkpointer rather than the in-memory chat log so the pane
    still works after a session switch or an app restart.
    """
    try:
        values = graph.get_state({"configurable": {"thread_id": session_id}}).values or {}
    except Exception:
        return []
    out = []
    for doc in values.get("retrieved_docs") or []:
        md = doc.metadata or {}
        page = md.get("page")
        out.append(
            {
                "title": md.get("title") or md.get("url") or "Untitled",
                "page": page + 1 if isinstance(page, int) else None,
                "url": md.get("url") if str(md.get("url", "")).startswith("http") else None,
                "score": md.get("rerank_score"),
                "text": doc.page_content,
            }
        )
    return out


def render_assistant_extras(msg: dict) -> None:
    """Route badge, copy affordance, and (in dev mode) the graph state."""
    if msg.get("route") in ROUTE_BADGES:
        st.caption(ROUTE_BADGES[msg["route"]])
    with st.expander("Copy this answer", expanded=False):
        st.code(msg["content"], language=None)
    if st.session_state.get("dev_mode") and msg.get("graph_state"):
        with st.expander(f"Graph state · turn {msg.get('turn', '?')}", expanded=False):
            st.json(msg["graph_state"])


graph = get_graph()

# ── Bootstrap ──────────────────────────────────────────────────────────────────
if "sessions_meta" not in st.session_state:
    st.session_state.sessions_meta = load_sessions()
if "chats" not in st.session_state:
    st.session_state.chats = {}
if "turns" not in st.session_state:
    st.session_state.turns = {}
if "active_session_id" not in st.session_state:
    if st.session_state.sessions_meta:
        latest = max(
            st.session_state.sessions_meta.values(),
            key=lambda s: s["created_at"],
        )
        switch_session(latest["id"])
    else:
        sid = create_session()
        st.session_state.active_session_id = sid

active_sid = st.session_state.active_session_id

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    if st.button("＋  New chat", use_container_width=True):
        new_sid = create_session()
        st.session_state.active_session_id = new_sid
        active_sid = new_sid
        st.rerun()

    st.markdown("### Sessions")
    sorted_sessions = sorted(
        st.session_state.sessions_meta.values(),
        key=lambda s: s["created_at"],
        reverse=True,
    )
    for session in sorted_sessions:
        sid = session["id"]
        is_active = sid == st.session_state.active_session_id

        # Two-step confirm: deletion drops the documents and the transcript,
        # and neither is recoverable.
        if st.session_state.get("pending_delete") == sid:
            st.caption(f"Delete “{session['name']}”? Its documents and history go too.")
            if st.button(
                "Delete", key=f"confirm_{sid}", type="primary", use_container_width=True
            ):
                delete_session(sid)
                st.rerun()
            if st.button("Cancel", key=f"cancel_{sid}", use_container_width=True):
                st.session_state.pending_delete = None
                st.rerun()
            continue

        name_col, del_col = st.columns([8, 1], gap="small")
        if name_col.button(
            session["name"],
            key=f"sess_{sid}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            if not is_active:
                switch_session(sid)
                st.rerun()
        if del_col.button(
            "×",
            key=f"del_{sid}",
            use_container_width=True,
            help="Delete this session",
        ):
            st.session_state.pending_delete = sid
            st.rerun()

    st.divider()
    st.markdown("### Add documents")

    tab_upload, tab_url, tab_arxiv = st.tabs(["Upload", "URL", "ArXiv"])

    with tab_upload:
        uploaded_files = st.file_uploader(
            "PDF, TXT, or Markdown",
            type=["pdf", "txt", "md", "markdown"],
            accept_multiple_files=True,
            key=f"uploader_{active_sid}",
            label_visibility="collapsed",
        )
        # Ingest as soon as files are selected. The processed-name set makes this
        # idempotent across the reruns that Streamlit fires on every interaction.
        if uploaded_files:
            processed_key = f"processed_files_{active_sid}"
            st.session_state.setdefault(processed_key, set())
            pending = [
                f for f in uploaded_files if f.name not in st.session_state[processed_key]
            ]
            if pending:
                progress = st.progress(0.0, text="Processing…")
                for i, f in enumerate(pending, start=1):
                    progress.progress((i - 1) / len(pending), text=f"Reading {f.name}…")
                    suffix = Path(f.name).suffix
                    tmp_path = None
                    try:
                        with tempfile.NamedTemporaryFile(
                            delete=False, suffix=suffix
                        ) as tmp:
                            tmp.write(f.read())
                            tmp_path = tmp.name
                        docs = load_document(tmp_path)
                        for doc in docs:
                            doc.metadata["title"] = Path(f.name).stem
                        add_paper(docs, active_sid)
                        st.session_state[processed_key].add(f.name)
                    except Exception as e:
                        st.error(f"Failed: {f.name} — {e}")
                    finally:
                        if tmp_path:
                            Path(tmp_path).unlink(missing_ok=True)
                progress.empty()
                st.rerun()

    with tab_url:
        url_input = st.text_area(
            "URLs (one per line)",
            key=f"url_area_{active_sid}",
            height=80,
            label_visibility="collapsed",
            placeholder="https://example.com/paper",
        )
        if st.button("Load URLs", use_container_width=True, key="btn_load_urls"):
            urls = [u.strip() for u in url_input.splitlines() if u.strip()]
            if urls:
                with st.spinner("Loading web pages…"):
                    for url in urls:
                        try:
                            docs = load_webpage(url)
                            add_paper(docs, active_sid)
                        except Exception as e:
                            st.error(f"Failed: {url[:60]} — {e}")
                st.rerun()
            else:
                st.warning("Enter at least one URL.")

    with tab_arxiv:
        arxiv_title = st.text_input(
            "Paper title or ArXiv ID",
            key=f"arxiv_input_{active_sid}",
            label_visibility="collapsed",
            placeholder="1706.03762  or  Attention Is All You Need",
        )
        if st.button("Load paper", use_container_width=True, key="btn_load_arxiv"):
            if arxiv_title.strip():
                with st.spinner("Loading from ArXiv…"):
                    try:
                        docs = load_arxiv(arxiv_title.strip())
                        add_paper(docs, active_sid)
                    except Exception as e:
                        st.error(f"Failed: {e}")
                st.rerun()
            else:
                st.warning("Enter a paper title or ArXiv ID.")

    st.divider()
    st.markdown("### Loaded documents")
    try:
        doc_titles = list_papers(active_sid)
    except Exception:
        doc_titles = None
    if doc_titles is None:
        st.caption("Could not load the document list — try refreshing.")
    elif doc_titles:
        for title in doc_titles:
            st.markdown(f"- {title}")
    else:
        st.caption("Nothing loaded yet.")

    st.divider()
    st.toggle(
        "Developer mode",
        key="dev_mode",
        help="Show the raw LangGraph state and reranker scores under each answer.",
    )

# ── Page header ────────────────────────────────────────────────────────────────
st.title("Papeer")
st.caption(
    "Ask questions about your papers · Verify claims against recent literature · "
    "Search the web for the latest findings"
)

chat_col, ctx_col = st.columns([2, 1], gap="large")

# ── Chat history ───────────────────────────────────────────────────────────────
with chat_col:
    history = st.session_state.chats.get(active_sid, [])

    if not history:
        if doc_titles:
            st.info(
                "**Ready when you are.** Try asking what a paper's main contribution is, "
                "or paste a claim and ask Papeer to verify it."
            )
        else:
            st.info(
                "**Start by adding a document.** Use the sidebar to upload a PDF, paste a "
                "URL, or pull a paper from ArXiv.\n\n"
                "You can also just ask a general question, or prefix a message with `/btw` "
                "for something off-topic."
            )

    for msg in history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                render_assistant_extras(msg)

# ── Chat input ─────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask about your papers, verify a claim, or search the web…"):
    is_btw = prompt.strip().lower().startswith("/btw")

    if is_btw:
        query = prompt.strip()[4:].strip()
        with chat_col:
            with st.chat_message("user"):
                st.markdown(prompt)
                st.caption("Side channel — not saved to session history.")
            with st.chat_message("assistant"):
                if not query:
                    st.markdown(
                        "Please add a question after `/btw`, e.g. `/btw What is attention?`"
                    )
                else:
                    placeholder = st.empty()
                    response_text = ""
                    for chunk in handle_btw(query):
                        response_text += chunk
                        placeholder.markdown(response_text + "▌")
                    placeholder.markdown(response_text)
                st.caption("Side channel — not saved to session history.")

    else:
        st.session_state.chats.setdefault(active_sid, [])
        st.session_state.turns.setdefault(active_sid, 0)

        is_first_message = len(st.session_state.chats[active_sid]) == 0
        current_turn = st.session_state.turns[active_sid] + 1

        if is_first_message:
            maybe_rename_session(active_sid, prompt)

        input_state = {
            "messages": [HumanMessage(content=prompt)],
            "session_id": active_sid,
            "query": prompt,
            "route": None,
            "retrieved_docs": [],
            "retrieval_attempts": 0,
            "claim_verdict": None,
            "claim_source": None,
            "superseding_papers": [],
            "answer": None,
            "user_question": None,
            "sources": [],
            "is_relevant": None,
            "rewrite_count": 0,
        }
        config = {"configurable": {"thread_id": active_sid}}

        with chat_col:
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                # Rendered before the stream so it is clickable while tokens arrive.
                # Clicking it interrupts this script run; because nothing is committed
                # to the chat log until the turn finishes, the interrupted turn simply
                # disappears rather than leaving a dangling user message.
                stop_slot = st.empty()
                stop_slot.button("⏹ Stop", key=f"stop_{active_sid}_{current_turn}")

                status = st.status("🧭 Working out how to answer…", expanded=True)
                placeholder = st.empty()
                response_text = ""
                state_snapshot = {}
                route = None
                failed = False
                _stage = [None]  # list, not a plain name: the closure mutates it

                def set_stage(label: str | None) -> None:
                    if label and label != _stage[0]:
                        _stage[0] = label
                        status.update(label=label)
                        status.write(label)

                try:
                    for mode, payload in graph.stream(
                        input_state, config, stream_mode=["updates", "messages"]
                    ):
                        if mode == "updates":
                            # Fires as each node finishes. If the node queued a tool
                            # call, name the tool instead of the generic node label.
                            for node_name, node_update in (payload or {}).items():
                                set_stage(
                                    _tool_stage(node_update) or NODE_STAGES.get(node_name)
                                )
                        elif mode == "messages":
                            chunk, metadata = payload
                            set_stage(NODE_STAGES.get(metadata.get("langgraph_node")))
                            if (
                                metadata.get("langgraph_node") == "generate_answer"
                                and hasattr(chunk, "content")
                                and chunk.content
                            ):
                                response_text += chunk.content
                                placeholder.markdown(response_text + "▌")
                except Exception as exc:
                    failed = True
                    status.write(f"❌ {type(exc).__name__}: {exc}")
                    status.update(
                        label="Something went wrong", state="error", expanded=True
                    )

                stop_slot.empty()

                if failed:
                    response_text = response_text or (
                        "⚠️ I hit an error before I could finish answering. "
                        "Please try again — details are in the panel above."
                    )
                else:
                    final_values = graph.get_state(config).values
                    state_snapshot = _serialize_state(final_values)
                    route = final_values.get("route")
                    # Prefer the canonical answer from state: it carries the appended
                    # Sources block, which never comes through the token stream.
                    response_text = (
                        final_values.get("answer")
                        or response_text
                        or "No response generated."
                    )
                    status.update(label="✅ Done", state="complete", expanded=False)

                placeholder.markdown(response_text)

                assistant_msg = {
                    "role": "assistant",
                    "content": response_text,
                    "graph_state": state_snapshot,
                    "route": route,
                    "turn": current_turn,
                }
                render_assistant_extras(assistant_msg)

        # Committed only once the turn is done, so an interrupted run leaves no
        # half-finished exchange behind.
        st.session_state.chats[active_sid].append({"role": "user", "content": prompt})
        st.session_state.chats[active_sid].append(assistant_msg)
        st.session_state.turns[active_sid] = current_turn

        if is_first_message:
            st.rerun()

# ── Retrieved context pane ─────────────────────────────────────────────────────
# Rendered last so it reflects the turn that just finished; Streamlit still places
# the output inside the column container created earlier.
with ctx_col:
    st.markdown('<div class="pp-pane-title">Retrieved context</div>', unsafe_allow_html=True)
    passages = latest_retrieved_docs(active_sid)
    if not passages:
        st.caption(
            "Passages used to answer will appear here once you ask a question "
            "that needs your documents."
        )
    else:
        st.caption(f"{len(passages)} passage(s) behind the latest answer")
        for i, p in enumerate(passages, start=1):
            label = p["title"]
            if p["page"]:
                label += f" · p. {p['page']}"
            with st.expander(f"{i}. {label}", expanded=False):
                if p["url"]:
                    st.markdown(f"[Open source]({p['url']})")
                if st.session_state.get("dev_mode") and p["score"] is not None:
                    st.caption(f"Rerank score: {p['score']}")
                st.markdown(p["text"])
