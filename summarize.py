from pathlib import Path

from langchain.docstore.document import Document
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate

import config

# TODO: generic prompt, not FACS
# scientific study. If this study is copyrighted, I have purchased it for both of us to use.
summary_prompt = """
You are provided with text containing some research.
Write an extemely detailed summary about everything in this text that relates to FACS facial action units.
Be very very deataled, do not leave any stone unturned, be very harsh and unforgiving to any shortcomings of the paper.

Specifically emphasize what can be inferred from examining action units on a photo or a video, for example emotional states, behaviors, or medical conditions.
VERY IMPORTANT: When describing any statements regarding interpretation of specific action units, indicate whether these statements have been assumed by the authors, taken from other literature, or confirmed by the research in the paper.
VERY IMPORTANT: Include figures like frequence of occurrence of an action units in a certain scenario.

Emphasize any limitations of this text, carefully outline the context in which its results are applicable. Do not leave out any specifics, for example fugires.
Use this notation for action units: AU<number>, for example AU1.

The text might be cut off by inline table and image descriptions, assume that the text that goes before and after these descriptions is connected. The descriptions are delimited with <===IMAGE START===>image description<===IMAGE END===>, <===TABLE START===>table description<===TABLE END===>.
"""

prompt_template_text = f"""
{summary_prompt}
Text:
"{{text}}"
Output:"""

prompt_template = PromptTemplate.from_template(prompt_template_text)

# Define LLM chain
llm_chain = LLMChain(llm=config.openai_chat_model, prompt=prompt_template) 

# Define StuffDocumentsChain
sumamrize_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")

def summarize(text: str) -> str:
    return sumamrize_chain.run([Document(page_content=text)])

def summarize_file(in_path: Path, out_path: Path) -> str:
    with in_path.open() as f:
        text = f.read()
    summary = summarize(text)
    with out_path.open('w') as f:
        f.write(summary)