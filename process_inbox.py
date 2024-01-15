import shutil
import time
import zipfile
from pathlib import Path
import concurrent.futures
from .pdf_parser.PdfProcessor import PdfProcessor
from .summarize import summarize_file
import sys

# from dotenv import load_dotenv, find_dotenv
# _ = load_dotenv(find_dotenv("mathpix.env"))


curpath = Path(__file__).parent.resolve()
inbox_path = curpath / 'inbox'
archive_path = curpath / 'archive'
library_path = curpath / 'library'
summaries_path = curpath / 'summaries'

def process_pdf(pdf_path: Path) -> str:
    print(f"Parsing PDF: {pdf_path}")

    processor = PdfProcessor()
    pdf_id = processor.get_mathpix_pdf_id(pdf_path)

    # Periodically call pdf_to_txt with a timeout of 1 minute
    # (it still can take arbitrary amount of time, it just won't restart if there's error after timeout)
    start_time = time.time()
    timeout = 60  # seconds

    while time.time() - start_time < timeout:
        try:
            text = processor.pdf_to_txt(pdf_id)
            with (library_path / f"{pdf_path.stem}.txt").open("w") as f:
                f.write(text)
            # Move the processed PDF to the 'archive' directory
            shutil.move(pdf_path, archive_path / pdf_path.name)
            break
        except zipfile.BadZipfile as e:
            time.sleep(5)  # seconds

    return f"{pdf_path.name}"


def process_directory(directory_path: Path):
    # Find all PDF files in the directory
    print('directory_path', directory_path)
    pdf_inbox = { p.name for p in directory_path.glob('*.pdf') if p.name[0] != '_' }
    pdf_parsed = { p.stem+'.txt' for p in library_path.glob('*.txt') }
    pdf_files = pdf_inbox - pdf_parsed
    print(f'parsing {len(pdf_files)} pdfs: ', pdf_files)

    # Use ProcessPoolExecutor for parallel processing
    # Since it's i/o bound, it can also be threads, and max_workers can be larger
    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
        # Submit each PDF file for processing in parallel
        future_to_path = { executor.submit(process_pdf, directory_path / pdf_path): pdf_path for pdf_path in pdf_files }

        # Wait for all tasks to complete and print progress
        for future in concurrent.futures.as_completed(future_to_path):
            pdf_path = future_to_path[future]
            try:
                print("Parsing task completed:", pdf_path)
            except Exception as e:
                print("Parsing task failed:", pdf_path)

    # TODO: summarize immediately and not wait for all the parses to finish
    parsed_pdfs = { p.name for p in library_path.glob('*.txt') if p.name[0] != '_' }
    summarized_pdfs = { p.name for p in summaries_path.glob('*.txt') }
    to_summarize = parsed_pdfs - summarized_pdfs
    print(f'summarizing {len(to_summarize)} pdfs...', to_summarize)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Submit each PDF file for processing in parallel
        future_to_path = { executor.submit(summarize_file, library_path/name, summaries_path/name): name for name in to_summarize }

        # Wait for all tasks to complete and print progress
        for future in concurrent.futures.as_completed(future_to_path):
            txt_path = future_to_path[future]
            try:
                print("Summarizing task completed:", txt_path)
            except Exception as e:
                print("Summarizing task failed:", txt_path)

if __name__ == "__main__":
    # Replace 'your_directory_path' with the actual path to your directory
    process_directory(Path(sys.argv[1] if len(sys.argv) > 1 else inbox_path))
