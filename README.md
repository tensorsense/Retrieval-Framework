# Retrieval Framework

This is a tool that converts scientific PDFs into plain text for your LLM-related needs.

- Convert PDF to LaTeX using [Mathpix API](https://docs.mathpix.com/#introduction) that is tailored to work with scientific papers.
- Extract images and tables from LaTeX and replace them with text using a multimodal LLM.
  - The prompts are made to extract all values and relationships represented within each table or graph and minimize information loss.

## Basic usage

Refer to [this notebook](https://github.com/tensorsense/Retrieval-Framework/blob/main/pdfs_to_rag.ipynb) to learn more about:

- Configuring and running end-to-end PDF conversion.
- Creating a basic LlamaIndex RAG on top of your converted documents
- Evaluating RAG performance using TruLens

1. Set `MATHPIX_APP_ID` and `MATHPIX_APP_KEY` in your environment. We suggest using a `.env` file.

```python
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv(".env"))  # read local .env file
```

2. Instantiate a text and a vision model. This tool uses LlamaIndex abstractions to interface with LLMs.

```python
from llama_index.llms import OpenAI
from llama_index.multi_modal_llms import OpenAIMultiModal

text_model = OpenAI()
vision_model = OpenAIMultiModal(max_new_tokens=4096)
```

Next, pass those models to the converter.

```python
converter = MathpixPdfConverter(text_model=text_model, vision_model=vision_model)
```

3. Convert PDF and extract the result.

```python
pdf_path = Path("path/to/file.pdf")

pdf_result = converter.convert(pdf_path)

with Path(f"output.txt").open("w") as f:
    f.write(pdf_result.content)
```

## Custom workflow

In order to persist intermediate results or run processing in parallel,
you can use `MathpixProcessor` and `MathpixResultParser` directly.

```python
processor = MathpixProcessor()
parser = MathpixResultParser(text_model=text_model, vision_model=vision_model)

mathpix_result = processor.submit_pdf(pdf_path)
mathpix_result = processor.await_result(mathpix_result)
pdf_result = parser.parse_result(mathpix_result)
```

## See also

- LlamaIndex [docs](https://docs.llamaindex.ai/en/stable/getting_started/starter_example.html)
- Mathpix API [docs](https://docs.mathpix.com/#introduction)
- TruLens [docs](https://www.trulens.org/trulens_eval/llama_index_quickstart/)
- Pydantic [docs](https://docs.pydantic.dev/latest/api/base_model/)