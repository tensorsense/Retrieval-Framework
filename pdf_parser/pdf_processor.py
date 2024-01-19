import base64
import io
import json
import os
import time
import zipfile
from pathlib import Path
from typing import List, Any

import requests
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from pylatexenc.latex2text import LatexNodes2Text
from tqdm import tqdm

from pdf_parser.latex_helpers import LatexChunk, LatexChunkType
from pdf_parser.latex_helpers import get_latex_chunks, fetch_img, fetch_tex_filename
from pdf_parser.latex_helpers import preprocess_regex, postprocess_regex
import logging

TABLE_TEMPLATE = \
    """Your role is to prepare scientific papers for blind scientists. You need to convert tables in scientific articles for their display on Braille linear displays. Attached is a certain table in LaTeX format. You need to:
    1.) Understand where the header is and where the content is.
    2.) Then, for each row, output the information in the following format: <Column 1 Name>: <value for column 1 for the current row>; <Column 2 Name>: <value for column 2 for the current row>; ...; <Last Column Name>: <value for the last column for the current row>
    3.) Each row of the table should be output on a separate line.
    4.) Do not skip lines, replace them with 'etc.', or similar abbreviations. All information from the table must be fully preserved.
    5.) If the header consists of more than one line, then the name of each column is formed from the vertical combination of all header cells for that column. The combination is made through a comma.
    6.) If the header has merged cells that extend to 2 or more columns - the value of this merged cell must be added to each vertical column and not only for the first in this merge.
    7.) Do not display any additional information except the result of performing the task according to the described algorithm. It is necessary to output the result immediately, otherwise, it will break the Braille monitor. NO INTROS. JUST RESULT.
    8.) If first column doesn't have a name, then write values for it without  colon.
    9.) Remove all TeX or LaTeX syntax in the output, use only understandable to the layman math symbols and notations.

    Here is the table:
    {table}
    """

IMAGE_PROMPT = \
    """
    Your role is to prepare scientific papers for blind scientists. You need to understand images in scientific articles for their text interpretation on Braille linear displays. Attached is a certain image from a scientific paper. You need to:
    1.) Understand what is in front of you - a plot/graph/diagram, a photo, or a schematic.
    2.) For a plot/graph/diagram, you need to provide a description of its purpose (if a description text is present), the names or meaning of each axis/dimension, and VERY IMPORTANT estimate and reproduce all the specific tabular values on which this image was created with repeated name of the metric together with each value itself.
    3.) For a photo, you need to describe what is depicted.
    4.) For a schematic, you need to describe all its elements and the connections between them.
    5.) For other images that do not fit into any of the three aforementioned categories, you need to describe in maximum detail what is depicted as if you are describing the image to an artist who needs to reproduce it in maximum detail, and another artist-critic will then compare and look for the smallest differences.
    6.) Do not skip lines, replace them with 'etc.', or similar abbreviations. All information from the image must be fully preserved.
    7.) Do not display any additional intro information except the result of performing the task according to the described algorithm. It is necessary to output the result immediately, otherwise, it will break the Braille monitor. NO INTROS. JUST DEDSCRIPTIVE RESULT.
    """

IMAGE_START_DELIMITER = '\n<===IMAGE START===>\n'
IMAGE_END_DELIMITER = '\n<===IMAGE END===>\n'
TABLE_START_DELIMITER = '\n<===TABLE START===>\n'
TABLE_END_DELIMITER = '\n<===TABLE END===>\n'


class MathpixResult(BaseModel):
    src_path: Path = None
    pdf_id: str = None
    zip_bytes: bytes = None
    error: str = None
    error_info: Any = None


class PdfResult(BaseModel):
    raw_latex: str = None
    content: str = None
    text: List[LatexChunk] = Field(default_factory=list)
    tables: List[LatexChunk] = Field(default_factory=list)
    images: List[LatexChunk] = Field(default_factory=list)


class MathpixProcessor:
    MATHPIX_ENDPOINT = "https://api.mathpix.com/v3/pdf"  # no trailing slash

    def __init__(self):
        self.headers = {
            "app_id": os.environ["MATHPIX_APP_ID"],
            "app_key": os.environ["MATHPIX_APP_KEY"],
        }

        self.options = {
            "conversion_formats": {
                "md": False,
                "docx": False,
                "tex.zip": True,
                "html": False
            },
            # "math_inline_delimiters": ["$", "$"],
            # "rm_spaces": True
        }

    def submit_pdf(self, pdf_path: Path) -> MathpixResult:
        logging.info(f"Submitting {pdf_path.name}...")
        response = requests.post(
            self.MATHPIX_ENDPOINT,
            headers=self.headers,
            data={"options_json": json.dumps(self.options)},
            files={"file": pdf_path.open("rb")}
        )

        mathpix_result = MathpixResult(src_path=pdf_path)

        if response.ok:
            response_dict = json.loads(response.text.encode("utf8"))
            mathpix_result.pdf_id = response_dict["pdf_id"]
            logging.info(f"Received PDF id: {mathpix_result.pdf_id}")
            if "error" in response_dict:
                logging.error(f"Receieved error from server: {response_dict['error']}")
                mathpix_result.error = response_dict["error"]
                mathpix_result.error_info = response_dict["error_info"]
        else:
            try:
                response.raise_for_status()
            except requests.exceptions as e:
                mathpix_result.error = e

        return mathpix_result

    def await_result(self, mathpix_result: MathpixResult, timeout_s: int = 60, sleep_s: int = 5) -> MathpixResult:
        with tqdm(total=100, desc="Processing...") as pbar:
            start_time = time.time()
            while True:
                if time.time() - start_time > timeout_s:
                    mathpix_result.error = f"Processing has exceeded {timeout_s}s limit."
                    break

                url = f"{self.MATHPIX_ENDPOINT}/{mathpix_result.pdf_id}"
                response = requests.get(url, headers=self.headers)

                if response.ok:
                    response_dict = response.json()
                    status = response_dict["status"]
                    percent_done = int(response_dict["percent_done"]) if "percent_done" in response_dict else 0

                    if status == "error":
                        logging.error("An error occurred during processing on the server side.")
                        mathpix_result.error = "An error occurred during processing on the server side."
                        break

                    pbar.update(percent_done - pbar.n)
                    if status == "completed":
                        logging.info("Completed processing")
                        break
                else:
                    try:
                        response.raise_for_status()
                    except requests.exceptions as e:
                        mathpix_result.error = e

                time.sleep(sleep_s)

        if mathpix_result.error:
            return mathpix_result

        try:
            logging.info("Downloading tex.zip...")
            url = f"{self.MATHPIX_ENDPOINT}/{mathpix_result.pdf_id}.tex"
            response = requests.get(url, headers=self.headers)
            mathpix_result.zip_bytes = response.content
        except requests.exceptions as e:
            logging.error("Error downloading tex.zip")
            mathpix_result.error = e

        return mathpix_result


class MathpixResultParser:
    def __init__(self, text_model: BaseChatModel, vision_model: BaseChatModel):
        self.text_model: BaseChatModel = text_model
        self.vision_model: BaseChatModel = vision_model

    def parse_result(self, mathpix_result: MathpixResult) -> PdfResult:
        assert mathpix_result.zip_bytes is not None, \
            f"Missing tex.zip content. Did you call MathpixProcessor.await_result()?"

        zip_buffer = io.BytesIO(mathpix_result.zip_bytes)
        zip_ref = zipfile.ZipFile(zip_buffer, "r")
        tex_filename = fetch_tex_filename(zip_ref)
        assert tex_filename is not None, \
            f"Could not find .tex file in tex.zip"

        pdf_result = PdfResult()
        with zip_ref.open(tex_filename, 'r') as tex_file:
            pdf_result.raw_latex = tex_file.read().decode('utf-8')

        logging.info("Processing latex...")
        latex_chunks = get_latex_chunks(pdf_result.raw_latex)
        for chunk in tqdm(latex_chunks, total=len(latex_chunks)):
            match chunk.type:
                case LatexChunkType.text:
                    chunk.processed_content = chunk.raw_content
                    pdf_result.text.append(chunk)
                case LatexChunkType.table:
                    chunk.processed_content = (f"{TABLE_START_DELIMITER}"
                                               f"\n{self.convert_table(chunk.raw_content)}"
                                               f"\n{TABLE_END_DELIMITER}")
                    pdf_result.tables.append(chunk)
                case LatexChunkType.image:
                    img_path = Path(f"{mathpix_result.pdf_id}") / "images" / f"{chunk.filename}.jpg"
                    img = fetch_img(zip_ref, img_path)
                    chunk.processed_content = (f"{IMAGE_START_DELIMITER}"
                                               f"\n{self.convert_image(img)}"
                                               f"\n{IMAGE_END_DELIMITER}")
                    pdf_result.images.append(chunk)
                case _:
                    raise ValueError(f"Unknown chunk type: {chunk.type}")

        logging.info("Cleaning up latex string")
        latex_str = "".join([chunk.processed_content for chunk in latex_chunks])
        latex_str = preprocess_regex(latex_str)
        latex_str = LatexNodes2Text().latex_to_text(latex_str)
        latex_str = postprocess_regex(latex_str)

        pdf_result.content = latex_str
        return pdf_result

    def convert_image(self, img_bytes: bytes) -> str:
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        img_url = f"data:image/jpeg;base64,{img_b64}"

        chain = self.vision_model | StrOutputParser()

        logging.info("Converting image...")
        response = chain.invoke(
            [
                HumanMessage(
                    content=[
                        {"type": "text", "text": IMAGE_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": img_url,
                                "detail": "auto",
                            },
                        },
                    ]
                )
            ]
        )
        return response

    def convert_table(self, table_str: str) -> str:
        prompt = ChatPromptTemplate.from_template(TABLE_TEMPLATE)
        chain = prompt | self.text_model | StrOutputParser()

        logging.info("Converting table...")
        response = chain.invoke({"table": table_str})
        return response


class MathpixPdfConverter:
    def __init__(self, text_model: BaseChatModel, vision_model: BaseChatModel):
        self.processor = MathpixProcessor()
        self.parser = MathpixResultParser(text_model=text_model, vision_model=vision_model)

    def convert(self, pdf_path: Path):
        mathpix_result = self.processor.submit_pdf(pdf_path)
        mathpix_result = self.processor.await_result(mathpix_result)
        pdf_result = self.parser.parse_result(mathpix_result)
        return pdf_result
