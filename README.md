# Retrieval Framework

## Configuration

1. Create `.env` from `.env.example`, provide:
- to use regular openai api (openai.com) provide `OPENAI_API_KEY`
- to use azure deployment provide:
  - `AZURE_OPENAI_ENDPOINT`
  - `AZURE_OPENAI_API_KEY`
  - `AZURE_OPENAI_DEPLOYMENT_CHAT`: required for PDF import and RAG
  - `AZURE_OPENAI_DEPLOYMENT_EMBEDDING`: required only for RAG
  - `AZURE_OPENAI_DEPLOYMENT_VISION`: required only for PDF import
- `MATHPIX_APP_ID` and `MATHPIX_APP_KEY`: required only for PDF import

2. In `config.py`:
- set `OPENAI_PROVIDER` according to .env: `openai` / `azure`

## PDF import

1. Put PDF files in `library/inbox`.
2. Run `python3 process_inbox.py`
3. Check `library/fulltext` for full text parses and `library/summary` for summaries.

Processed PDFs are moved to `library/archive`.

## Build the RAG
Build a RAG chain based on a simple retriever on top of the text library.
Run `python3 factory.py`

## Evaluate the RAG

Calculate a [RAG triad](https://www.trulens.org/trulens_eval/core_concepts_rag_triad/) of metrics using TruLens.

1. set `TRULENS_QUESTIONS` in config.py
2. Run `python3 eval.py`.
3. Open the dashboard in a browser at localhost:8000.
4. Wait for evaluation runs to finish.
