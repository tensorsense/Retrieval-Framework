import zipfile
import re
from enum import Enum, auto
from pathlib import Path
from typing import List

from pydantic import BaseModel
from pylatexenc.latexwalker import LatexWalker, LatexEnvironmentNode, LatexMacroNode, LatexGroupNode, LatexNode


class LatexChunkType(Enum):
    table = auto()
    text = auto()
    image = auto()


class LatexChunk(BaseModel):
    start: int  # starting position of a chunk within latex string
    end: int  # ending position
    type: LatexChunkType  # type of content inside the chunk
    filename: str = None  # filename to which the node links, if applicable
    raw_content: str = None  # piece of a latex string from start to end
    processed_content: str = None  # content after processing


def fetch_tex_filename(zip_ref: zipfile.ZipFile) -> str:
    """
    Fetches the name of the source latex file from the archive
    """
    tex_filename = None
    for file in zip_ref.namelist():
        if file.endswith(".tex"):
            tex_filename = file
            break
    return tex_filename


def fetch_img(zip_ref: zipfile.ZipFile, path: Path) -> bytes:
    """
    Extracts the image from the archive
    """
    with zip_ref.open(path.as_posix(), "r") as f:
        img = f.read()
    return img


def get_latex_chunks(latex_str: str) -> List[LatexChunk]:
    """
    Splits latex string into a list of LatexChunk objects, each containing a piece of regular text,
    a latex table or a latex code that inserts an image.
    :param latex_str: Raw latex string
    :return: List of LatexChunk, each with .raw_content. Images additionally get .filename.
    """
    walker = LatexWalker(latex_str)
    nodes, _, _ = walker.get_latex_nodes()

    def extract_chunks(nodes: List[LatexNode]) -> List[LatexChunk]:
        chunks = []
        for node in nodes:
            if isinstance(node, LatexEnvironmentNode) and node.environmentname == "tabular":
                chunks.append(
                    LatexChunk(
                        start=node.pos,
                        end=node.pos + node.len,
                        type=LatexChunkType.table,
                    )
                )
            elif isinstance(node, LatexMacroNode) and node.macroname == "includegraphics":
                image_name = node.nodeargs[0].nodelist[0].chars  # this is where the filename is
                chunks.append(
                    LatexChunk(
                        start=node.pos,
                        end=node.pos + node.len,
                        type=LatexChunkType.image,
                        filename=image_name
                    )
                )
            elif isinstance(node, LatexGroupNode) or isinstance(node, LatexEnvironmentNode):
                chunks.extend(extract_chunks(node.nodelist))
        return chunks

    # 1. Extract chunks that are registered as table or image nodes
    table_image_chunks = extract_chunks(nodes)

    # 2. Extract regular text chunks that are located in between text and image nodes
    all_chunks = []
    last_pos = 0
    for chunk in table_image_chunks:
        if last_pos < chunk.start:
            all_chunks.append(
                LatexChunk(
                    start=last_pos,
                    end=chunk.start,
                    type=LatexChunkType.text
                )
            )
        last_pos = chunk.end
        # Make sure the table/image chunk is in all_chunks as well
        all_chunks.append(chunk)

    # Don't forget the one at the end
    if last_pos < len(latex_str):
        all_chunks.append(
            LatexChunk(
                start=last_pos,
                end=len(latex_str),
                type=LatexChunkType.text,
            )
        )

    # 3. For each chunk, extract it's corresponding piece of latex string
    for chunk in all_chunks:
        chunk.raw_content = latex_str[chunk.start:chunk.end]

    return all_chunks


def preprocess_regex(latex_string):
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


def postprocess_regex(latex_string):
    latex_string = re.sub(r'^\s+', '', latex_string, flags=re.MULTILINE)
    latex_string = re.sub(r'(\n\s*){2,}', '\n\n', latex_string)
    return latex_string
