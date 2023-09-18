import PyPDF2
import typing

# --------------------------------------------------------------------------
# _BasicKits naive Writers, aliases
# --------------------------------------------------------------------------

from .._BasicKits.FileT import list_to_file, base64_to_file, write_dict_to_csv


# --------------------------------------------------------------------------
# Non-Naive functions
# --------------------------------------------------------------------------

def pdf_extract_partof(pdf_read_path, partofpdf_save_path,
                       start_index: int = 0, end_index: typing.Optional[int] = None):
    # Extract pages from 'from_page' to 'to_page'
    with open(pdf_read_path, 'rb') as f:
        # Create a PDF file reader object
        # pdf_reader = PyPDF2.PdfFileReader(f)
        pdf_reader = PyPDF2.PdfReader(f)

        # Create a PDF writer object to save the extracted pages
        pdf_writer = PyPDF2.PdfFileWriter()

        # Loop through each page in the PDF and add the result_text to the string
        end_index = end_index if end_index else len(pdf_reader.pages)

        for page_num in range(start_index, end_index):
            # page = pdf_reader.getPage(page_num)
            page = pdf_reader.pages[page_num]
            # result_text += page.extractText()
            pdf_writer.addPage(page=page)

        with open(partofpdf_save_path, 'wb') as temp_pdf_file:
            pdf_writer.write(temp_pdf_file)
