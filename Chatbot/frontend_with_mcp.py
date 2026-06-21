import queue
import uuid

import streamlit as st
from backend_with_mcp import chatbot, retrieve_all_threads, submit_async_task
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

# =========================== Utilities ===========================

def generate_thread_id():
    # Keep this a string so it always matches the thread_ids coming back
    # from retrieve_all_threads() (sqlite/checkpointer stores them as strings).
    return str(uuid.uuid4())


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])


def messages_to_history(messages):
    """Convert LangChain messages into the {role, content} dicts used for
    rendering, skipping ToolMessages and empty AI turns (e.g. pure tool-call
    steps with no final text) so the UI only shows real conversation turns."""
    history = []
    for msg in messages:
        if isinstance(msg, HumanMessage) and msg.content:
            history.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage) and msg.content:
            history.append({"role": "assistant", "content": msg.content})
        # ToolMessage and empty AIMessage (tool-call-only) steps are skipped
    return history


def thread_label(thread_id):
    """Friendly sidebar label: first user message, truncated."""
    try:
        for msg in load_conversation(thread_id):
            if isinstance(msg, HumanMessage) and msg.content:
                text = msg.content.strip().replace("\n", " ")
                return text[:40] + ("…" if len(text) > 40 else "")
    except Exception:
        pass
    return str(thread_id)[:8]


# ======================= Session Initialization ===================

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

add_thread(st.session_state["thread_id"])

# ============================ Sidebar ============================

st.sidebar.title("LangGraph MCP Chatbot")

if st.sidebar.button("➕ New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

st.sidebar.header("My Conversations")
for thread_id in st.session_state["chat_threads"][::-1]:
    label = thread_label(thread_id)
    is_active = thread_id == st.session_state["thread_id"]
    button_label = f"🟢 {label}" if is_active else label

    if st.sidebar.button(button_label, key=f"thread_{thread_id}", use_container_width=True):
        st.session_state["thread_id"] = thread_id
        messages = load_conversation(thread_id)
        st.session_state["message_history"] = messages_to_history(messages)
        st.rerun()

# ============================ Main UI ============================

# Render history
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Type here")

if user_input:
    # Show user's message
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {"thread_id": st.session_state["thread_id"]},
        "run_name": "chat_turn",
    }

    # Assistant streaming block
    with st.chat_message("assistant"):
        status_holder = {"box": None, "tools_used": set()}
        stream_error = {"exc": None}

        def ai_only_stream():
            event_queue: queue.Queue = queue.Queue()

            async def run_stream():
                try:
                    async for message_chunk, metadata in chatbot.astream(
                        {"messages": [HumanMessage(content=user_input)]},
                        config=CONFIG,
                        stream_mode="messages",
                    ):
                        event_queue.put((message_chunk, metadata))
                except Exception as exc:
                    event_queue.put(("__error__", exc))
                finally:
                    event_queue.put(None)

            submit_async_task(run_stream())

            while True:
                item = event_queue.get()
                if item is None:
                    break

                message_chunk, metadata = item

                if message_chunk == "__error__":
                    stream_error["exc"] = metadata
                    break

                # Track every distinct tool used, update the status box live
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    status_holder["tools_used"].add(tool_name)
                    label = "🔧 Using " + ", ".join(
                        f"`{t}`" for t in sorted(status_holder["tools_used"])
                    )
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(label, expanded=True)
                    else:
                        status_holder["box"].update(label=label, state="running", expanded=True)

                # Stream ONLY assistant text tokens
                if isinstance(message_chunk, AIMessage) and message_chunk.content:
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())

        if stream_error["exc"] is not None:
            st.error(f"⚠️ Something went wrong: {stream_error['exc']}")
            ai_message = ai_message or "⚠️ The response was interrupted due to an error."

        if status_holder["box"] is not None:
            status_holder["box"].update(label="✅ Tool finished", state="complete", expanded=False)

        # write_stream can return "" if the model only made tool calls with
        # no final text reply (shouldn't normally happen, but guard anyway)
        if not ai_message:
            ai_message = "_(no response generated)_"
            st.markdown(ai_message)

    # Save assistant message
    st.session_state["message_history"].append({"role": "assistant", "content": ai_message})