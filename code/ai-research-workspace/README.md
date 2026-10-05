# 🔬 AI Research Workspace

An NLP-based research assistant built with **Streamlit, LangChain, OpenAI, and FAISS**.
Upload research-paper PDFs, then chat with them, summarize them, compare them, find
research gaps, and extract key concepts. Answers are grounded in the uploaded papers
and cite the **filename + page**.

## Features

| Page | What it does |
|------|--------------|
| 📄 Documents | Upload multiple PDFs; text is extracted page by page, cleaned, and chunked with metadata |
| 💬 Research Chat | RAG question answering with FAISS retrieval, source citations, and chat history |
| 📝 Summarize | Structured summary: Problem, Objective, Methodology, Dataset, Results, Contributions, Limitations, Future Work |
| 🔍 Compare | Compares two papers on methodology, datasets, models, results, and limitations |
| 🧠 Research Gaps | Limitations, unexplored areas, and research directions (clearly labelled as AI-generated observations) |
| 🔑 Key Concepts | Keywords, algorithms, datasets, metrics, and research concepts |

If the answer is not in the uploaded papers, the assistant says so instead of guessing.

## How it works

```text
PDF upload -> pypdf text extraction (per page) -> cleaning
          -> RecursiveCharacterTextSplitter (chunks keep filename + page)
          -> OpenAI embeddings -> in-memory FAISS index
Question  -> FAISS top-k retrieval -> prompt (context + question) -> LLM
          -> answer with [n] citations -> sources shown as filename + page
```

Summaries, comparison, gap detection, and concept extraction send the paper text
(truncated to fit the context window; see `config.py`) to the LLM using the prompts in `prompts.py`.

## Project structure

```text
ai-research-workspace/
├── app.py                # Streamlit UI and page routing
├── config.py             # Model, chunk size, overlap, top-k, temperature, limits
├── prompts.py            # All prompts (RAG, summary, comparison, gaps, concepts)
├── document_processor.py # PDF extraction, cleaning, chunking, metadata
├── rag.py                # Embeddings, FAISS, grounded QA, shared LLM helper
├── summarizer.py         # Summaries and concept extraction
├── comparator.py         # Paper comparison and research-gap detection
├── utils.py              # File size formatting, chat export
├── requirements.txt
├── .env.example
└── .gitignore
```

## Setup (Windows, Python 3.11)

```bat
cd ai-research-workspace
py -3.11 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Open `.env` in a text editor and replace `your_key_here` with your OpenAI API key
(create one at https://platform.openai.com/api-keys). Then run:

```bat
streamlit run app.py
```

The app opens at http://localhost:8501.

On macOS/Linux, use `python3.11 -m venv venv`, `source venv/bin/activate`, and `cp .env.example .env`.

> The app starts even without an API key. You can upload PDFs, but chat and analysis
> features will show a friendly message until the key is added.

## Configuration

Edit `config.py` (or override the models in `.env`):

| Setting | Default | Meaning |
|---------|---------|---------|
| `OPENAI_MODEL` | `gpt-4o-mini` | Chat model (env: `OPENAI_MODEL`) |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model (env: `OPENAI_EMBEDDING_MODEL`) |
| `TEMPERATURE` | `0.2` | Low = more factual |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `TOP_K` | `6` | Chunks retrieved per question |

## Demo walkthrough

1. **Documents**: upload 2-3 papers and click **Process documents**.
2. **Research Chat**: ask "What dataset was used?" and open **Sources** to see filename + page.
3. Ask something unrelated (for example "Who won the 2018 World Cup?") to show the "not found" behavior.
4. **Summarize**: generate a structured summary and download it.
5. **Compare**: pick two papers.
6. **Research Gaps** and **Key Concepts**: run them on all papers.

## Limitations

- Scanned (image-only) PDFs are not supported because there is no OCR.
- Very long papers are truncated (middle omitted) for summaries, comparison, gaps, and concepts. Chat search covers the whole paper.
- The index and chat history live in memory and are lost when the browser session ends (no database, by design).
- Two-column layouts and equations may extract imperfectly with pypdf.
- LLM output can contain mistakes; verify important claims against the paper.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| "OPENAI_API_KEY is missing" | Create `.env` from `.env.example` and add your key |
| Authentication error | The key is wrong or revoked; generate a new one |
| Rate limit / quota error | Check billing and usage on the OpenAI dashboard |
| Model not found | Set `OPENAI_MODEL` in `.env` to a model your account can use |
| `faiss` install fails | Use Python 3.11 (64-bit) and run `pip install --upgrade pip` first |
| "No extractable text" | The PDF is scanned; use a text-based PDF |

## Tech stack

Python 3.11 · Streamlit · LangChain · OpenAI API · FAISS · pypdf · python-dotenv