# Travel Guide Chatbot

An AI-powered travel assistant that answers destination questions using a full RAG (Retrieval-Augmented Generation) pipeline. Ask about beaches, food, transport, itineraries, and more across 8 global destinations.

Built by [Shangeet Sankar](https://www.linkedin.com/in/shangeet-sankar/) | [GitHub](https://github.com/ShangeetVishnuSankar)

---

## What it does

- Answers natural language travel questions grounded in real travel content
- Retrieves the most relevant information from a knowledge base before generating a response
- Remembers recent conversation turns so follow-up questions work naturally
- Cites the sources used in every answer

---

## Tech Stack

| Component | Tool |
|---|---|
| LLM | Gemini 2.5 Flash |
| Embeddings | Gemini Embedding 1 |
| Vector Database | Pinecone |
| Framework | LangChain |
| UI | Streamlit |

---

## Destinations in the Knowledge Base

Bali, Tokyo, Paris, Dubai, New York City, Doha, Bangkok, Istanbul

---

## Project Structure

```
travel-guide-chatbot/
├── app.py                    # Streamlit UI
├── src/
│   ├── data_collector.py     # Scrapes Wikivoyage pages
│   ├── document_loader.py    # Loads text and PDF files into LangChain Documents
│   ├── text_splitter.py      # Splits documents into chunks
│   ├── vector_store.py       # Embeds chunks and uploads to Pinecone
│   └── rag_chain.py          # Retriever + memory + LLM chain
├── data/
│   ├── web/                  # Scraped destination text files
│   ├── tips/                 # Manually written travel tips
│   └── pdfs/                 # Optional PDF travel guides
├── requirements.txt
└── .env                      # API keys (not committed)
```

---

## Running Locally

**1. Clone the repo and set up a virtual environment**
```bash
git clone https://github.com/ShangeetVishnuSankar/travel-guide-chatbot.git
cd travel-guide-chatbot
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**2. Add your API keys**

Create a `.env` file in the project root:
```
GOOGLE_API_KEY=your-google-api-key
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_INDEX_NAME=travel-guide
```

Get your Google API key at [aistudio.google.com](https://aistudio.google.com/apikey).
Get your Pinecone API key at [pinecone.io](https://www.pinecone.io).

**3. Collect data and build the vector store**
```bash
python src/data_collector.py
python src/vector_store.py --ingest
```

**4. Run the app**
```bash
python -m streamlit run app.py
```

---

## How RAG Works in this Project

```
User question
      |
      v
Conversation memory (last 3 turns)
      |
      v
LLM condenses follow-up into a standalone question
      |
      v
Pinecone retrieves the most relevant chunks
      |
      v
Gemini generates an answer grounded in those chunks
      |
      v
Answer + cited sources shown in UI
```

---

## API Keys Required

- **Google AI Studio** (free) -- covers both Gemini LLM and Gemini Embedding
- **Pinecone** (free tier) -- vector database
