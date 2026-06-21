import streamlit as st
from backend_with_db import chatbot, retrieve_all_threads
from langchain_core.messages import HumanMessage
import uuid

# **************************************** utility functions *************************

def generate_thread_id():
    # Use str(...) so this always matches the string thread_ids coming back
    # from retrieve_all_threads() (sqlite stores them as strings).
    return str(uuid.uuid4())


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(thread_id)
    st.session_state['message_history'] = []


def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)


def load_conversation(thread_id):
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    # Check if messages key exists in state values, return empty list if not
    return state.values.get('messages', [])


def thread_label(thread_id):
    """Use the first user message as a friendly label for the thread,
    falling back to the raw thread_id if no messages exist yet."""
    try:
        messages = load_conversation(thread_id)
        for msg in messages:
            if isinstance(msg, HumanMessage) and msg.content:
                text = msg.content.strip().replace("\n", " ")
                return text[:40] + ("..." if len(text) > 40 else "")
    except Exception:
        pass
    return str(thread_id)[:8]  # short fallback


# **************************************** Session Setup ******************************

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrieve_all_threads()

add_thread(st.session_state['thread_id'])

# **************************************** Sidebar UI *********************************

st.sidebar.title('LangGraph Chatbot')

if st.sidebar.button('➕ New Chat', use_container_width=True):
    reset_chat()
    st.rerun()

st.sidebar.header('My Conversations')

for thread_id in st.session_state['chat_threads'][::-1]:
    label = thread_label(thread_id)
    is_active = thread_id == st.session_state['thread_id']
    button_label = f"🟢 {label}" if is_active else label

    if st.sidebar.button(button_label, key=f"thread_{thread_id}", use_container_width=True):
        st.session_state['thread_id'] = thread_id
        messages = load_conversation(thread_id)

        temp_messages = []
        for msg in messages:
            role = 'user' if isinstance(msg, HumanMessage) else 'assistant'
            temp_messages.append({'role': role, 'content': msg.content})

        st.session_state['message_history'] = temp_messages
        st.rerun()

# **************************************** Main UI ************************************

# loading the conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

user_input = st.chat_input('Type here')

if user_input:
    # first add the message to message_history
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.markdown(user_input)

    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {"thread_id": st.session_state["thread_id"]},
        "run_name": "chat_turn",
    }

    with st.chat_message('assistant'):
        try:
            ai_message = st.write_stream(
                message_chunk.content for message_chunk, metadata in chatbot.stream(
                    {'messages': [HumanMessage(content=user_input)]},
                    config=CONFIG,
                    stream_mode='messages'
                )
            )
        except Exception as e:
            ai_message = f"⚠️ Something went wrong: {e}"
            st.error(ai_message)

    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})