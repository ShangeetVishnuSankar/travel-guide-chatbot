"""
RAG CHAIN
=========
The heart of the application — connects retriever, prompt, and LLM.

CONCEPTS YOU'LL LEARN:
- Retrievers (wrapper around vector store search)
- Prompt engineering for RAG
- LangChain chains (LCEL — LangChain Expression Language)
- Combining retrieved context with user queries
- Source citation in responses
- Conversation memory with question condensation
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from vector_store import load_vector_store

load_dotenv()

# -----------------------------------------------------------------
# MEMORY CONFIGURATION
# -----------------------------------------------------------------
# How many conversation turns (user + assistant pairs) to remember.
# 3 means the bot remembers the last 3 back-and-forth exchanges.
# More turns = more context but slightly more API usage per call.
MEMORY_TURNS = 3


# -----------------------------------------------------------------
# STEP 1: RETRIEVER
# -----------------------------------------------------------------
def get_retriever(search_k: int = 4):
    """
    Create a retriever from the vector store.

    A RETRIEVER is simply a wrapper around the vector store that:
    - Takes a query string
    - Returns the top-k most similar documents

    SEARCH TYPES YOU COULD EXPLORE:
    - "similarity"       → pure cosine similarity (default)
    - "mmr"              → Maximum Marginal Relevance (balances
                           relevance with diversity — avoids
                           returning 4 near-identical chunks)
    - "similarity_score_threshold" → only return chunks above
                                     a confidence threshold
    """
    vector_store = load_vector_store()

    retriever = vector_store.as_retriever(
        search_type="mmr",   # Try "similarity" too and compare!
        search_kwargs={
            "k": search_k,   # Number of chunks to retrieve
            "fetch_k": 20,   # MMR fetches more, then diversifies
        },
    )
    return retriever


# -----------------------------------------------------------------
# STEP 2: PROMPT TEMPLATES
# -----------------------------------------------------------------

# --- Condensation Prompt ---
# This prompt is used BEFORE retrieval.
# It takes the conversation history + follow-up question and rewrites
# the follow-up into a standalone question that Pinecone can search.
#
# WHY IS THIS NEEDED?
# If the user asks "Tell me about Bali" and then "What about the food?",
# Pinecone receives "What about the food?" — which is too vague to search.
# The condensation step rewrites it to "What is the food like in Bali?"
# so the retriever can find the right chunks.

CONDENSATION_PROMPT_TEMPLATE = """Given the conversation history and a follow-up question,
rewrite the follow-up into a standalone question that contains all the context needed
to search a travel knowledge base. If the follow-up is already standalone, return it unchanged.

CONVERSATION HISTORY:
{chat_history}

FOLLOW-UP QUESTION: {question}

STANDALONE QUESTION:"""

condensation_prompt = ChatPromptTemplate.from_template(CONDENSATION_PROMPT_TEMPLATE)


# --- RAG Answer Prompt ---
RAG_PROMPT_TEMPLATE = """You are a friendly, knowledgeable travel guide assistant.
Your job is to help users plan trips and answer travel questions using ONLY
the context provided below. You have deep expertise in destinations worldwide.

RULES:
1. Only answer based on the provided context. If the context doesn't contain
   enough information, say so honestly and suggest what the user could search for.
2. Always mention which destination(s) you're drawing information from.
3. Be specific — include names of places, neighborhoods, price ranges when available.
4. If the user asks about a destination not in your knowledge base, let them know.
5. Keep responses conversational and helpful, like a friend who's been there.

CONTEXT (retrieved from travel knowledge base):
{context}

USER QUESTION: {question}

YOUR ANSWER:"""

rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)


# -----------------------------------------------------------------
# STEP 3: LLM
# -----------------------------------------------------------------
def get_llm():
    """
    Initialize Google Gemini 2.5 Flash as the language model.

    This is the "brain" that reads the retrieved context + question
    and generates a natural language answer.

    WHY GEMINI 2.5 FLASH?
    - Free tier: 5 RPM, 25 TPM (enough for development & demos)
    - Fast inference (it's the "Flash" variant)
    - Strong reasoning for its size
    - Same API key you already have for embeddings

    ALTERNATIVES:
    - gemini-2.5-pro   → better quality, lower free-tier limits
    - gemini-2.0-flash → previous generation, higher free limits
    """
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,  # Lower = more factual, less creative
    )


# -----------------------------------------------------------------
# STEP 4: BUILD THE CHAIN (LangChain Expression Language — LCEL)
# -----------------------------------------------------------------
def format_chat_history(messages: list[dict]) -> str:
    """
    Format the last MEMORY_TURNS exchanges into a readable string
    for the condensation prompt.

    Each message is a dict with "role" (user/assistant) and "content" keys —
    the same format that st.session_state.messages uses in app.py.

    MEMORY_TURNS = 3 means we take the last 6 messages (3 user + 3 assistant).
    """
    if not messages:
        return "No previous conversation."

    # Take the last MEMORY_TURNS * 2 messages (each turn = 1 user + 1 assistant)
    recent = messages[-(MEMORY_TURNS * 2):]

    formatted = []
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        # Truncate long assistant messages to keep the prompt lean
        content = msg["content"][:300] + "..." if len(msg["content"]) > 300 else msg["content"]
        formatted.append(f"{role}: {content}")

    return "\n".join(formatted)


def format_docs(docs):
    """Format retrieved documents into a single string for the prompt."""
    formatted = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "unknown")
        formatted.append(
            f"[Source {i + 1}: {source}]\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(formatted)


def create_rag_chain():
    """
    Create the full RAG chain using LCEL (LangChain Expression Language).

    THE CHAIN FLOW:
    1. User question comes in
    2. Retriever finds relevant chunks
    3. Chunks are formatted into a context string
    4. Prompt template combines context + question
    5. LLM generates the answer
    6. Output parser extracts the text

    LCEL uses the pipe operator (|) to chain steps:
      retriever | prompt | llm | output_parser
    """
    retriever = get_retriever()
    llm = get_llm()
    output_parser = StrOutputParser()

    # The chain
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | rag_prompt
        | llm
        | output_parser
    )

    return rag_chain


def create_rag_chain_with_sources():
    """
    Extended chain that also returns source documents and supports
    conversation memory via question condensation.

    CHAIN FLOW (with memory):
    1. Receive current question + last N turns of chat history
    2. If history exists → LLM condenses into a standalone question
    3. Condensed question → Pinecone retrieval
    4. Retrieved context + condensed question → LLM generates answer
    5. Return answer + sources

    CHAIN FLOW (first message, no history):
    1. Receive current question (no history yet)
    2. Skip condensation — question is already standalone
    3. Question → Pinecone retrieval
    4. Retrieved context + question → LLM generates answer
    5. Return answer + sources
    """
    retriever = get_retriever()
    llm = get_llm()
    output_parser = StrOutputParser()

    def run_chain(question: str, chat_history: list[dict] = None) -> dict:
        chat_history = chat_history or []

        # Step 1: Condense the question if there's prior conversation
        if chat_history:
            history_str = format_chat_history(chat_history)
            condensed_question = (condensation_prompt | llm | output_parser).invoke({
                "chat_history": history_str,
                "question": question,
            })
            print(f"  🔄 Condensed: '{question}' → '{condensed_question}'")
        else:
            # First message — no condensation needed
            condensed_question = question

        # Step 2: Retrieve using the (possibly condensed) question
        docs = retriever.invoke(condensed_question)
        context = format_docs(docs)
        sources = [
            {
                "source": doc.metadata.get("source", "unknown"),
                "content_preview": doc.page_content[:200],
            }
            for doc in docs
        ]

        # Step 3: Generate the answer
        chain = rag_prompt | llm | output_parser
        answer = chain.invoke({
            "context": context,
            "question": condensed_question,
        })

        return {
            "answer": answer,
            "sources": sources,
        }

    return run_chain


# --- TEST IT ---
if __name__ == "__main__":
    print("🔗 Building RAG chain...")
    chain = create_rag_chain_with_sources()

    # Test queries
    test_questions = [
        "What are the must-visit temples in Bali?",
        "How do I use the subway system in Tokyo?",
        "What's the best street food to try in Bangkok?",
        "Any tips for visiting the Eiffel Tower in Paris?",
        "What's the weather like in Dubai?",
    ]

    for question in test_questions:
        print(f"\n{'=' * 60}")
        print(f"❓ {question}")
        print(f"{'=' * 60}")

        result = chain(question)

        print(f"\n💬 {result['answer']}")
        print(f"\n📚 Sources used:")
        for src in result["sources"]:
            print(f"   - {src['source']}")
