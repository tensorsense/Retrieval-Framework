# Retrieval Framework

This is the RAG component of the Demo Agent.

## PDF import

1. Configure `MATHPIX_APP_ID`, `MATHPIX_APP_KEY` and `OPENAI_API_KEY` in `mathpix.env`.
2. Put PDF files in the inbox.
3. Run `python3 process_inbox.py`

Every PDF in the inbox is going to be transformed into a plain-text file. Original PDFs are stored in the archives.

- To transform a PDF, it is first converted to LaTeX using Mathpix API.
- Tables and images get parsed out of the resulting file and sent to OpenAI GPT-4 and GPT-4-Vision for summarization.

## Build the RAG

Build a RAG chain based on a simple retriever on top of the library.

In order to try:

1. Configure Azure credentials in `azure.env` and uncommend `dotenv` loading in `factory.py`.
2. Run `python3 factory.py`

## Evaluate RAG

Calculate a [RAG triad](https://www.trulens.org/trulens_eval/core_concepts_rag_triad/) of metrics using TruLens.

1. Configure Azure credentials as before.
2. Run `python3 eval.py`.
3. Open the dashboard at port 8000.
4. Wait for evaluation runs to finish. 