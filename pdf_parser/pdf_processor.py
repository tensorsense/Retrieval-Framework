import hashlib
import sqlite3
import os
import json
import zipfile
import io
import base64
from pathlib import Path
import re

import requests
from pylatexenc.latexwalker import LatexWalker, LatexEnvironmentNode, LatexMacroNode, LatexGroupNode
from pylatexenc.latex2text import LatexNodes2Text

import config

class PdfProcessor:
    IMAGE_START_DELIMITER = '\n<===IMAGE START===>\n'
    IMAGE_END_DELIMITER = '\n<===IMAGE END===>\n'
    TABLE_START_DELIMITER = '\n<===TABLE START===>\n'
    TABLE_END_DELIMITER = '\n<===TABLE END===>\n'

    def __init__(self, db_path=Path(__file__).parent.resolve()/'pdfs.db'):
        self.db_path = db_path
        self._initialize_db()
        self.headers = {
            "app_id": config.MATHPIX_APP_ID,
            "app_key": config.MATHPIX_APP_KEY,
        }
        self.result = []

    def _initialize_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS pdfs 
                          (md5 TEXT PRIMARY KEY, 
                           filename TEXT, 
                           mathpix_pdf_id TEXT)''')
        conn.commit()
        conn.close()


    def get_mathpix_pdf_id(self, pdf_path):
        md5_hash = self._calculate_md5(pdf_path)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('SELECT mathpix_pdf_id FROM pdfs WHERE md5 = ?', (md5_hash,))
        result = cursor.fetchone()

        if result:
            print(f'{pdf_path} {md5_hash} in db.')
            conn.close()
            return result[0]

        print(f'{pdf_path} {md5_hash} not in db, uploading to mathpix.')

        mathpix_pdf_id = self._process_pdf_with_mathpix(pdf_path)
        cursor.execute('INSERT INTO pdfs (md5, filename, mathpix_pdf_id) VALUES (?, ?, ?)',
                       (md5_hash, os.path.basename(pdf_path), mathpix_pdf_id))
        conn.commit()
        conn.close()
        return mathpix_pdf_id


    def _calculate_md5(self, file_path):
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()


    def _process_pdf_with_mathpix(self, pdf_path):
        options = {
            "conversion_formats": {
                "md": False,
                "docx": False,
                "tex.zip": True,
                "html": False
            },
            # "math_inline_delimiters": ["$", "$"],
            # "rm_spaces": True
        }


        r = requests.post(
                            "https://api.mathpix.com/v3/pdf",
                            headers=self.headers,
                            data={ "options_json": json.dumps(options) },
                            files={ "file": open(pdf_path, "rb") }
                         )

        if r.ok:
            response = r.text.encode("utf8")
            return json.loads(response)['pdf_id']
        else:
            return None


    def pdf_to_txt(self, pdf_id):
        url = "https://api.mathpix.com/v3/pdf/" + pdf_id + ".tex"
        response = requests.get(url, headers=self.headers)

        zip_file = io.BytesIO(response.content)

        zip_ref = zipfile.ZipFile(zip_file, 'r')

        tex_filename = None
        for file in zip_ref.namelist():
            if file.endswith(".tex"):
                tex_filename = file
                break

        if not tex_filename:
            return "No .tex file found in the zip archive."

        with zip_ref.open(tex_filename, 'r') as tex_file:
            latex_string = tex_file.read().decode('utf-8')

        walker = LatexWalker(latex_string)
        nodes, _, _ = walker.get_latex_nodes()
        pieces = self.find_latex_pieces(nodes)

        last_pos = 0
        text_pieces = []
        for start, length, type, filename in pieces:
            if last_pos < start:
                text_pieces.append((last_pos, start - last_pos, 'text', None))
            text_pieces.append((start, length, type, filename))
            last_pos = start + length
        if last_pos < len(latex_string):
            text_pieces.append((last_pos, len(latex_string) - last_pos, 'text', None))

        processed_latex_string = ''
        for start, length, type, filename in text_pieces:
            #print(start)
            fragment = latex_string[start:start + length]
            if type == 'text':
                processed_latex_string += fragment
            elif type == 'image':
                img = self._read_image_from_zip(pdf_id, filename, zip_ref)
                processed_latex_string += self.IMAGE_START_DELIMITER
                processed_latex_string += self._process_image(img)
                processed_latex_string += self.IMAGE_END_DELIMITER
            elif type == 'table':
                processed_latex_string += self.TABLE_START_DELIMITER
                processed_latex_string += self._process_table(fragment)
                processed_latex_string += self.TABLE_END_DELIMITER

        processed_latex_string = self.regex_preprocessing(processed_latex_string)

        processed_latex_string = LatexNodes2Text().latex_to_text(processed_latex_string)

        processed_latex_string = self.regex_postprocessing(processed_latex_string)

        zip_ref.close()

        return processed_latex_string


    def regex_preprocessing(self, latex_string):
        # Delete tags \urlstyle{...}
        latex_string = re.sub(r'\\urlstyle{.*?}', '', latex_string)

        # Delete tags \graphicspath{...}
        latex_string = re.sub(r'\\graphicspath{.*?}', '', latex_string)

        # Replace tags \href{URL}{text} на text
        def replace_href(match):
            try:
                return match.group(2) + " (" + match.group(1) + ") "
            except:
                return match.group(1)

        latex_string = re.sub(r'\\href{.*?}{(.*?)}', replace_href, latex_string)

        return latex_string

    def regex_postprocessing(self, latex_string):
        latex_string = re.sub(r'^\s+', '', latex_string, flags=re.MULTILINE)
        latex_string = re.sub(r'(\n\s*){2,}', '\n\n', latex_string)
        return latex_string


    def find_latex_pieces(self, nodes):
        pieces = []

        for node in nodes:
            if isinstance(node, LatexEnvironmentNode) and node.environmentname == 'tabular':
                pieces.append((node.pos, node.len, 'table', None))
            elif isinstance(node, LatexMacroNode) and node.macroname == 'includegraphics':
                image_name = self._extract_image_path_from_node(node)
                pieces.append((node.pos, node.len, 'image', image_name))
            elif isinstance(node, LatexGroupNode) or isinstance(node, LatexEnvironmentNode):
                pieces += self.find_latex_pieces(node.nodelist)

        return pieces


    def _extract_text_from_tex(self, pdf_id, tex_content, zip_ref):
        lw = LatexWalker(tex_content)
        nodes, _, _ = lw.get_latex_nodes()
        self.result = []
        processed_text = self._process_nodes(pdf_id, tex_content, nodes, zip_ref)
        return self.result

    def _convert_node_to_latex(self, node, tex_content):
        start, end = node.pos, node.pos + node.len
        return tex_content[start:end]

    def _process_nodes(self, pdf_id, tex_content, nodes, zip_ref):

        for node in nodes:
            if isinstance(node, LatexEnvironmentNode) and node.environmentname == 'tabular':
                table_text = self._convert_node_to_latex(node, tex_content)
                processed_table = self._process_table(table_text)
                self.result.append(processed_table)
            elif isinstance(node, LatexMacroNode):
                if node.macroname == 'includegraphics':
                    image_name = self._extract_image_path_from_node(node)
                    image_content = self._read_image_from_zip(pdf_id, image_name, zip_ref)
                    processed_image = self._process_image(image_content)
                    self.result.append(processed_image)
                else:
                    pass
            elif isinstance(node, LatexGroupNode) or isinstance(node, LatexEnvironmentNode):
                group_text = self._process_nodes(pdf_id, tex_content, node.nodelist, zip_ref)
            else:
                node_latex = self._convert_node_to_latex(node, tex_content)
                text = self._latex_to_text(node_latex)
                if text:
                    self.result.append(text)


    def _extract_image_path_from_node(self, node):
        if isinstance(node, LatexMacroNode) and node.macroname == 'includegraphics':
            return node.nodeargs[0].nodelist[0].chars
        return None

    def _read_image_from_zip(self, pdf_id, image_filename, zip_ref):
        image_path = f"{pdf_id}/images/{image_filename}.jpg"
        with zip_ref.open(image_path, 'r') as image_file:
            image_content = image_file.read()
        return image_content

    def _process_table(self, table_text):
        try:
            response = config.openai_model.chat.completions.create(
                messages=[
                    {"role": "user", "content": process_table_prompt + table_text}
                ],
            )
            result = response.choices[0].message.content
            return result
        except Exception as e:
            print(f"Error processing table: {e}")
            raise e

    def _process_image(self, image_content):
        try:
            base64_image = base64.b64encode(image_content).decode('utf-8')

            response = config.openai_vision_model.chat.completions.create(
                messages=[
                    {"role": "user", "content": [
                        {"type": "text", "text": process_image_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]}
                ],
                temperature=0,
                max_tokens=4096
            )

            result = response.choices[0].message.content
            return result
        except Exception as e:
            print(f"Error processing image: {e}")
            raise e



process_table_prompt = \
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

process_image_prompt = \
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