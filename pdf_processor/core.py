import base64
import hashlib
import io
import json
import os
import time
import zipfile
from pathlib import Path
from typing import List, Any, Optional

import requests
from llama_index.core.llms.types import ChatMessage
from llama_index.multi_modal_llms.openai_utils import generate_openai_multi_modal_chat_message
from llama_index.llms import LLM
from llama_index.multi_modal_llms import MultiModalLLM
from llama_index.schema import ImageDocument

from pydantic import BaseModel, Field
from pylatexenc.latex2text import LatexNodes2Text
from tqdm import tqdm

from pdf_processor.latex_helpers import LatexChunk, LatexChunkType
from pdf_processor.latex_helpers import get_latex_chunks, fetch_img, fetch_tex_filename
from pdf_processor.latex_helpers import preprocess_regex, postprocess_regex
import logging

TABLE_PROMPT = \
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
    src_path: Optional[Path] = None  # path to original pdf
    pdf_id: Optional[str] = None  # id assigned by Mathpix API
    zip_b64: Optional[str] = None  # b64 encoded tex.zip file with processing results
    zip_hash: Optional[str] = None  # sha256 hash of the raw tex.zip
    error: Optional[str] = None  # an error encountered during processing on Mathpix side or on the app side
    error_info: Optional[Any] = None


class PdfResult(BaseModel):
    raw_latex: Optional[str] = None  # original latex string extracted from the .tex file
    content: Optional[str] = None  # text after conversion and cleanup
    text: List[LatexChunk] = Field(default_factory=list)
    tables: List[LatexChunk] = Field(default_factory=list)  # pre-conversion chunks extracted from raw latex
    images: List[LatexChunk] = Field(default_factory=list)


class MathpixProcessor:
    """
    A class that interacts with Mathpix API from uploading files to downloading results.
    """
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
        """
        Submit a PDF file to Mathpix.

        :param pdf_path: pathlib.Path to file
        :return: MathpixResult object with .pdf_id (and NO result)

        Does not raise errors but stores them in the returned object.
        """
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

    def await_result(
            self,
            mathpix_result: MathpixResult,
            timeout_s: int = 60,
            sleep_s: int = 5
    ) -> MathpixResult:
        """
        Request processing status in a loop, download results when complete.

        :param mathpix_result: Mathpix result object WITH .pdf_id
        :param timeout_s: max wait time in seconds
        :param sleep_s: interval between status requests
        :return: Mathpix result object with .zip_b64 (if download is successful)
        """
        if mathpix_result.error is not None:
            logging.warning(f"Received MathpixResult with non-empty error: {mathpix_result.error}. Ignoring...")
            mathpix_result.error = None

        with tqdm(total=100, desc="Processing on the Mathpix side...") as pbar:
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
            mathpix_result.zip_hash = hashlib.sha256(response.content).hexdigest()
            mathpix_result.zip_b64 = base64.b64encode(response.content).decode("utf-8")
        except requests.exceptions as e:
            logging.error("Error downloading tex.zip")
            mathpix_result.error = e

        return mathpix_result


class MathpixResultParser:
    """
    This class is for extracting information from a .tex file and converting in to text.
    """

    def __init__(self, text_model: LLM, vision_model: MultiModalLLM):
        """
        :param text_model: instance of a LlamaIndex LLM such as OpenAI
        :param vision_model: instance of a LlamaIndex MultiModalLLM such as OpenAIMultiModal
        """
        self.text_model: LLM = text_model
        self.vision_model: MultiModalLLM = vision_model

    def parse_result(self, mathpix_result: MathpixResult) -> PdfResult:
        """
        :param mathpix_result: MathpixResult object with .pdf_id and .zip_b64.
            Could be from MathpixProcessor or made manually.
        :return: PdfResult object with .content (fully processed text) and intermediate results
        """
        assert mathpix_result.zip_b64 is not None, \
            f"Missing tex.zip content. Did you call MathpixProcessor.await_result()?"

        zip_bytes = base64.b64decode(mathpix_result.zip_b64.encode("utf-8"))
        assert hashlib.sha256(zip_bytes).hexdigest() == mathpix_result.zip_hash, "tex.zip got broken during storage"

        zip_buffer = io.BytesIO(zip_bytes)
        zip_ref = zipfile.ZipFile(zip_buffer, "r")
        tex_filename = fetch_tex_filename(zip_ref)
        assert tex_filename is not None, \
            f"Could not find .tex file in tex.zip"

        pdf_result = PdfResult()
        with zip_ref.open(tex_filename, 'r') as tex_file:
            pdf_result.raw_latex = tex_file.read().decode('utf-8')

        logging.info("Processing latex...")
        latex_chunks = get_latex_chunks(pdf_result.raw_latex)
        for chunk in tqdm(latex_chunks, total=len(latex_chunks), desc="Converting latex chunks..."):
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
                    chunk.file_b64 = base64.b64encode(img).decode("utf-8")
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
        """
        Run image through a vision model to extract information into text
        """
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        img_doc = ImageDocument(image=img_b64, image_mimetype="image/jpeg")
        message = generate_openai_multi_modal_chat_message(
            prompt=IMAGE_PROMPT,
            role="user",
            image_documents=[img_doc],
        )
        logging.info("Converting image...")
        response = self.vision_model.chat([message])
        return response.message.content

    def convert_table(self, table_str: str) -> str:
        """
        Run latex table through an LLM to extract information into text
        """
        response = self.text_model.chat(
            [
                ChatMessage(role="user", content=f"{TABLE_PROMPT}\n{table_str}"),
            ]
        )
        return response.message.content


class MathpixProcessingError(BaseException):
    pass


class MathpixPdfConverter:
    """
    Convenience class for processing PDFs using MathPix API
    """

    def __init__(self, text_model: LLM, vision_model: MultiModalLLM):
        self.processor = MathpixProcessor()
        self.parser = MathpixResultParser(text_model=text_model, vision_model=vision_model)

    def convert(self, pdf_path: Path) -> PdfResult:
        """
        Convert pdf to text
        :param pdf_path: Path to the pdf
        :return: PdfResult with .content attribute that contains final text
        """
        mathpix_result = self.processor.submit_pdf(pdf_path)
        if mathpix_result.error:
            raise MathpixProcessingError(mathpix_result.error)
        mathpix_result = self.processor.await_result(mathpix_result)
        if mathpix_result.error:
            raise MathpixProcessingError(mathpix_result.error)
        pdf_result = self.parser.parse_result(mathpix_result)
        return pdf_result
