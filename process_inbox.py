import shutil
import time
import zipfile
from pathlib import Path
import concurrent.futures
import sys
from functools import partial

from pdf_processor.pdf_processor import PdfProcessor
from summarize import summarize_file
import config

library_path = Path(__file__).parent.resolve() / 'library'
inbox_path = library_path / 'inbox'
archive_path = library_path / 'archive'
fulltext_path = library_path / 'fulltext'
summary_path = library_path / 'summary'

def process_pdf(pdf_path: Path, summarize=True) -> str:
    print(f"Parsing PDF: {pdf_path}")

    processor = PdfProcessor()
    pdf_id = processor.get_mathpix_pdf_id(pdf_path)

    # Periodically call pdf_to_txt with a timeout of 1 minute
    # (it still can take arbitrary amount of time, it just won't restart if there's error after timeout)
    start_time = time.time()
    timeout = config.PDF_PARSE_TIMEOUT  # seconds

    while time.time() - start_time < timeout:
        try:
            text = processor.pdf_to_txt(pdf_id)
            with (fulltext_path / f"{pdf_path.stem}.txt").open("w") as f:
                f.write(text)
            # Move the processed PDF to the 'archive' directory
            shutil.move(pdf_path, archive_path / pdf_path.name)
            break
        except zipfile.BadZipfile as e:
            time.sleep(5)  # seconds

    if summarize:
        summarize_file(fulltext_path / f"{pdf_path.stem}.txt", summary_path / f"{pdf_path.stem}.txt")

    return f"{pdf_path.name}"


def process_directory(summarize=True):
    # Find all PDF files in the directory
    print('directory_path', library_path)
    pdf_inbox = { p.name for p in library_path.glob('*.pdf') if p.name[0] != '_' }
    pdf_parsed = { p.stem+'.txt' for p in fulltext_path.glob('*.txt') }
    pdf_files = pdf_inbox - pdf_parsed
    print(f'parsing {len(pdf_files)} pdfs: ', pdf_files)

    # Use ProcessPoolExecutor for parallel processing
    # Since it's i/o bound, it can also be threads, and max_workers can be larger
    with concurrent.futures.ProcessPoolExecutor(max_workers=config.parse_pdfs_num_workers) as executor:
        # Submit each PDF file for processing in parallel
        future_to_path = { executor.submit(process_pdf, library_path / pdf_path, summarize): pdf_path for pdf_path in pdf_files }

        # Wait for all tasks to complete and print progress
        for future in concurrent.futures.as_completed(future_to_path):
            pdf_path = future_to_path[future]
            try:
                print("Parsing task completed:", pdf_path)
            except Exception as e:
                print("Parsing task failed:", pdf_path)


def summarize_directory():
    "summarize all pdfs that dont have summaries"

    parsed_pdfs = { p.name for p in fulltext_path.glob('*.txt') if p.name[0] != '_' }
    summarized_pdfs = { p.name for p in summary_path.glob('*.txt') }
    to_summarize = parsed_pdfs - summarized_pdfs
    print(f'summarizing {len(to_summarize)} pdfs...', to_summarize)

    with concurrent.futures.ProcessPoolExecutor(max_workers=config.summarize_pdfs_num_workers) as executor:
        # Submit each PDF file for processing in parallel
        future_to_path = { executor.submit(summarize_file, fulltext_path/name, summary_path/name): name for name in to_summarize }

        # Wait for all tasks to complete and print progress
        for future in concurrent.futures.as_completed(future_to_path):
            txt_path = future_to_path[future]
            try:
                print("Summarizing task completed:", txt_path)
            except Exception as e:
                print("Summarizing task failed:", txt_path)

if __name__ == "__main__":
    # Replace 'your_directory_path' with the actual path to your directory
    process_directory()
