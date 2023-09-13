import html2text
import re
import bs4
import os
import PyPDF2
import pypandoc
import warnings
import typing
from .. import _BasicKits


def __sep_letter_warning():
    warnings.warn(
        """
        The converter names like 'convert_<...>_to_single_line_str' you using may occurs problems like:
        l o w i n g the salary w i l l m a k e t h e worker a r g u i n g t h e c o m p a n y.
        IF you want to correct this problem,
        you have to use nlptoolkits/SmallKits/WordninjaT/replace_sequence_letters_to_words_str
        """
    )


def convert_html_to_single_line_str(html_filepath, strike_tags: list = ["s", "strike", "del"], suppress_warn=False):
    """
    Args:
        html_filepath: file path
        strike_tags: the tags of strike html tags, default ["s", "strike", "del"]

    Returns: flat string with no \s{2,}, no \n, \t etc include, but only \s

    """
    if suppress_warn:
        __sep_letter_warning()

    encodetype = _BasicKits._BasicFuncT.find_file_encoding(html_filepath) \
        if not _BasicKits._BasicFuncT.find_file_encoding(html_filepath) is None else 'utf-8'
    # Open the HTML file and read it into a string
    with open(html_filepath, 'r', encoding=encodetype, errors='replace') as f:
        html_content = f.read()

    # Parse the HTML with BeautifulSoup
    soup = bs4.BeautifulSoup(html_content, 'html.parser')

    # Find and remove all strikethrough text
    for strike_tag in soup(strike_tags):
        strike_tag.decompose()

    # Convert the modified HTML to text
    h = html2text.HTML2Text()
    h.ignore_links = True
    result_text = h.handle(str(soup))
    # remove all \s+ to ' '
    result_text = re.sub(r'\s+', ' ', result_text, flags=re.IGNORECASE)

    return result_text


def convert_pdf_to_single_line_str(pdf_file_path, start_index: int = 0, end_index: typing.Optional[int] = None,
                                   suppress_warn=False
                                   ):
    """
    Args:
        pdf_file_path: pdf file path
        start_index: the page index to start reading
        end_index: None default, however if not None must be int, the end index
    Returns: flat string with no \s{2,}, no \n, \t etc include, but only \s

    """
    if suppress_warn:
        __sep_letter_warning()
    # Open the PDF file in binary mode
    with open(pdf_file_path, 'rb') as f:
        # Create a PDF file reader object
        # pdf_reader = PyPDF2.PdfFileReader(f)
        pdf_reader = PyPDF2.PdfReader(f)

        # Initialize an empty string to store the result_text
        result_text = ''

        # Loop through each page in the PDF and add the result_text to the string
        end_index = end_index if end_index else len(pdf_reader.pages)

        # for page_num in range(pdf_reader.getNumPages()):
        for page_num in range(start_index, end_index):
            # page = pdf_reader.getPage(page_num)
            page = pdf_reader.pages[page_num]
            # result_text += page.extractText()
            result_text += page.extract_text()

    # Remove newline characters to make the result_text a single line
    result_text = re.sub(r'\s+', ' ', result_text, flags=re.IGNORECASE)

    return result_text


def convert_doc_to_single_line_str(doc_file_path, temp_txt_file_path, suppress_warn=False):
    """
    Args:
        doc_file_path: read doc file path, need antiword engine
        temp_txt_file_path: the path to save temp txt file

    Returns: flat string with no \s{2,}, no \n, \t etc include, but only \s

    """
    if suppress_warn:
        __sep_letter_warning()
    # converting .doc to .docx
    doc_file = doc_file_path

    os.system('antiword ' + doc_file + ' > ' + temp_txt_file_path)

    encodetype = _BasicKits._BasicFuncT.find_file_encoding(temp_txt_file_path) \
        if not _BasicKits._BasicFuncT.find_file_encoding(temp_txt_file_path) is None else 'utf-8'

    with open(temp_txt_file_path, 'r',
              encoding=encodetype,
              errors='replace') as file:
        result_text = file.read()

    # Remove newline characters to make the result_text a single line
    result_text = re.sub(r'\s+', ' ', result_text, flags=re.IGNORECASE)

    return result_text


def convert_rtf_to_single_line_str(rtf_file_path, suppress_warn=False):
    """
    Args:
        rtf_file_path: the rtf file path

    Returns: flat string with no \s{2,}, no \n, \t etc include, but only \s

    """
    if suppress_warn:
        __sep_letter_warning()
    # Convert the RTF file to TXT format
    result_text = pypandoc.convert_file(rtf_file_path, 'plain', format='rtf')

    # Remove newline characters to make the text a single line
    result_text = re.sub(r'\s+', ' ', result_text, flags=re.IGNORECASE)

    return result_text
