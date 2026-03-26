"""
TRAVEL GUIDE CHATBOT — Streamlit App
=====================================
A user-friendly chat interface for the RAG-powered travel assistant.

CONCEPTS YOU'LL LEARN:
- Streamlit chat interface (st.chat_message, st.chat_input)
- Session state management (conversation history)
- Displaying sources/citations alongside answers
- Sidebar for configuration
- Custom CSS for modern UI styling
"""

import sys
import streamlit as st

# Add src to path so we can import our modules
sys.path.insert(0, "src")

from rag_chain import create_rag_chain_with_sources


# -----------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------
st.set_page_config(
    page_title="Travel Guide Chatbot",
    page_icon="🌍",
    layout="wide",
)


# -----------------------------------------------------------------
# CUSTOM CSS — modern, clean styling
# -----------------------------------------------------------------
st.markdown("""
<style>
    /* Import a clean modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main background */
    .stApp {
        background-color: #0f1117;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #161b27;
        border-right: 1px solid #2a2f3e;
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        background-color: #1a1f2e;
        border-radius: 12px;
        padding: 4px;
        margin-bottom: 8px;
        border: 1px solid #2a2f3e;
    }

    /* Input box */
    [data-testid="stChatInput"] {
        border-radius: 12px;
        border: 1px solid #2a2f3e;
        background-color: #1a1f2e;
    }

    /* Expander (sources) */
    [data-testid="stExpander"] {
        background-color: #1a1f2e;
        border: 1px solid #2a2f3e;
        border-radius: 8px;
    }

    /* Divider */
    hr {
        border-color: #2a2f3e;
    }

    /* Footer credit bar */
    .footer-bar {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: #161b27;
        border-top: 1px solid #2a2f3e;
        padding: 8px 24px;
        display: flex;
        align-items: center;
        justify-content: flex-end;
        gap: 20px;
        font-size: 12px;
        color: #6b7280;
        z-index: 999;
    }

    .footer-bar a {
        color: #6b7280;
        text-decoration: none;
        transition: color 0.2s;
    }

    .footer-bar a:hover {
        color: #9ca3af;
    }

    .footer-bar .separator {
        color: #2a2f3e;
    }

    /* Header subtitle */
    .header-sub {
        color: #6b7280;
        font-size: 14px;
        margin-top: -12px;
        margin-bottom: 20px;
    }

    /* Welcome card shown before first message */
    .welcome-card {
        background-color: #1a1f2e;
        border: 1px solid #2a2f3e;
        border-radius: 16px;
        padding: 32px 40px;
        text-align: center;
        margin: 60px auto;
        max-width: 560px;
    }

    .welcome-card h2 {
        font-size: 24px;
        font-weight: 600;
        margin-bottom: 8px;
        color: #f9fafb;
    }

    .welcome-card p {
        color: #6b7280;
        font-size: 14px;
        line-height: 1.6;
        margin-bottom: 20px;
    }

    .chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        justify-content: center;
    }

    .chip {
        background-color: #0f1117;
        border: 1px solid #2a2f3e;
        border-radius: 20px;
        padding: 6px 14px;
        font-size: 13px;
        color: #9ca3af;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🌍 Travel Guide")
    st.markdown("""
<div style='margin-top:-8px; margin-bottom:16px;'>
    <div style='color:#6b7280; font-size:13px; margin-bottom:8px;'>AI-powered trip planning</div>
    <div style='font-size:13px; color:#9ca3af;'>
        Shangeet Sankar &nbsp;·&nbsp;
        <a href='https://github.com/ShangeetVishnuSankar' target='_blank'
           style='color:#9ca3af; text-decoration:none;'>GitHub</a>
        &nbsp;·&nbsp;
        <a href='https://www.linkedin.com/in/shangeet-sankar/' target='_blank'
           style='color:#9ca3af; text-decoration:none;'>LinkedIn</a>
    </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
### How it works
1. **You ask** a travel question
2. **Retriever** finds relevant info from the knowledge base
3. **LLM** generates a natural answer using that info
4. **Sources** are shown so you know where the info came from
5. **Chat memory** keeps track of recent conversation so follow-up questions work naturally
""")

    st.divider()

    st.markdown("""
### 📚 Knowledge Base
- Bali
- Tokyo
- Paris
- Dubai
- New York City
- Doha
- Bangkok
- Istanbul
""")

    st.divider()

    st.markdown("""
### ⚙️ Tech Stack
- **LLM:** Gemini 2.5 Flash
- **Embeddings:** Gemini Embedding 1
- **Vector DB:** Pinecone
- **Memory:** Conversational RAG
- **Framework:** LangChain + Streamlit
""")

    st.divider()

    # Show/hide sources toggle
    show_sources = st.toggle("Show source documents", value=True)

    # Clear chat button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# -----------------------------------------------------------------
# INITIALIZE SESSION STATE
# -----------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_chain" not in st.session_state:
    with st.spinner("🔧 Loading travel knowledge base..."):
        st.session_state.rag_chain = create_rag_chain_with_sources()


# -----------------------------------------------------------------
# MAIN CHAT AREA — Header
# -----------------------------------------------------------------
st.markdown("## 🌍 Travel Guide Chatbot")
st.markdown('<div class="header-sub">Ask me anything about travel destinations — powered by Gemini + RAG.</div>', unsafe_allow_html=True)


# -----------------------------------------------------------------
# WELCOME CARD — shown only before the first message
# -----------------------------------------------------------------
if not st.session_state.messages:
    st.markdown("""
<div class="welcome-card">
    <h2>Where to next? ✈️</h2>
    <p>Ask me about destinations, things to do, local food, transport,<br>or anything else travel-related.</p>
    <div class="chip-row">
        <span class="chip">🏝️ Beaches in Bali</span>
        <span class="chip">🍜 Street food in Bangkok</span>
        <span class="chip">🚇 Tokyo subway tips</span>
        <span class="chip">🗼 Paris in 3 days</span>
        <span class="chip">🌆 Dubai on a budget</span>
    </div>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------
# CHAT HISTORY
# -----------------------------------------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Show sources for assistant messages
        if message["role"] == "assistant" and show_sources and "sources" in message:
            with st.expander("📚 View Sources"):
                for src in message["sources"]:
                    st.markdown(f"**{src['source']}**")
                    st.caption(src["content_preview"])
                    st.divider()


# -----------------------------------------------------------------
# HANDLE USER INPUT
# -----------------------------------------------------------------
if prompt := st.chat_input("Where do you want to explore? ✈️"):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching travel knowledge base..."):
            # Pass all messages except the one we just added (the current question)
            # The chain uses these to condense follow-up questions into standalone ones
            chat_history = st.session_state.messages[:-1]
            result = st.session_state.rag_chain(prompt, chat_history=chat_history)

        st.markdown(result["answer"])

        # Show sources
        if show_sources and result["sources"]:
            with st.expander("📚 View Sources"):
                for src in result["sources"]:
                    st.markdown(f"**{src['source']}**")
                    st.caption(src["content_preview"])
                    st.divider()

    # Save to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
    })


