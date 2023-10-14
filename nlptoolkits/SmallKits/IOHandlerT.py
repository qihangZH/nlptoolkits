import html2text
import re
import PyPDF2
import pypandoc
import warnings
import typing
import subprocess
import tempfile
import bs4
import os
import warnings
import readability

from .. import _BasicKits

# --------------------------------------------------------------------------
# _BasicKits naive readers, aliases
# --------------------------------------------------------------------------

from .._BasicKits.FileT import file_to_list

# --------------------------------------------------------------------------
# _BasicKits naive Writers, aliases
# --------------------------------------------------------------------------

from .._BasicKits.FileT import list_to_file, base64_to_file, write_dict_to_csv


# --------------------------------------------------------------------------
# GROBID-NEI-PDF->XML reader
# --------------------------------------------------------------------------
def _tei_xml_to_list(tei_xml_path,
                     tag: typing.Union[list, str],
                     subtags: typing.Optional[typing.Union[list, str]],
                     errors='backslashreplace',
                     **kwargs
                     ):
    """
    read tei-format XML to list of text
    Args:
        tei_xml_path: the path of Tei-xml-path, is a format could be seen in https://tei-c.org/ for detail
        tag: the tag which contain text you interested in, first level, should be ['text'] in default
        subtags: second level tags, like p/s/head, etc

    Returns: a list of text

    """
    if isinstance(tag, list):
        if len(tag)>1:
            warnings.warn('More then one tag will cause replicated result, you should known that')
    elif isinstance(tag, str):
        tag = [tag]
    else:
        raise ValueError('The tag either be string of list')

    encodetype = _BasicKits._BasicFuncT.find_file_encoding(tei_xml_path) \
        if not _BasicKits._BasicFuncT.find_file_encoding(tei_xml_path) is None else 'utf-8'

    with open(tei_xml_path, 'r', encoding=encodetype, errors=errors) as tei:
        soup = bs4.BeautifulSoup(tei, 'xml')

    # Initialize a list to store <s> pairs from all <div> elements
    s_pairs_list = []
    for t in tag:
        div_elements = soup.find_all(t)

        for div in div_elements:
            if subtags:
                for st in subtags:
                    s_elements = div.find_all(st)
                    sentences = [s.get_text(strip=True) for s in s_elements]
                    s_pairs_list.extend(sentences)
            else:
                sentences = div.get_text(strip=True)
                s_pairs_list.append(sentences)

    return s_pairs_list


# --------------------------------------------------------------------------
# Basic functions
# --------------------------------------------------------------------------

def __sep_letter_warning():
    warnings.warn(
        """
        The converter names like 'convert_<...>_to_single_line_str' you using may occurs problems like:
        l o w i n g the salary w i l l m a k e t h e worker a r g u i n g t h e c o m p a n y.
        IF you want to correct this problem,
        you have to use nlptoolkits/SmallKits/WordninjaT/replace_sequence_letters_to_words_str
        """
    )


# --------------------------------------------------------------------------
# Readers, ..._to_single_line_str series
# --------------------------------------------------------------------------

def convert_teixml_to_single_line_str(tei_xml_path,
                                      tag: typing.Union[list, str] = ['text'],
                                      subtags: typing.Optional[typing.Union[list, str]] = None,
                                      errors='backslashreplace',
                                      sep: str = ' ',
                                      suppress_warn: bool = False,
                                      **kwargs
                                      ):
    if not suppress_warn:
        __sep_letter_warning()

    string_list = _tei_xml_to_list(tei_xml_path,
                                   tag=tag,
                                   subtags=subtags,
                                   errors=errors,
                                   **kwargs
                                   )
    result_text = sep.join(string_list)
    # remove all \s+ to ' '
    result_text = re.sub(r'\s+', ' ', result_text, flags=re.IGNORECASE)

    return result_text


def convert_html_to_single_line_str(html_filepath, strike_tags: list = ["s", "strike", "del"],
                                    html_partial=False, suppress_warn=False, errors='backslashreplace', **kwargs):
    """
    Args:
        html_filepath: file path
        strike_tags: the tags of strike html tags, default ["s", "strike", "del"]
        html_partial: return only the div of the document, don't wrap
                             in html and body tags. see readability.Document().summary
        suppress_warn: is or not suppress the warn of sep letter


    Returns: flat string with no \s{2,}, no \n, \t etc include, but only \s

    """
    if not suppress_warn:
        __sep_letter_warning()

    encodetype = _BasicKits._BasicFuncT.find_file_encoding(html_filepath) \
        if not _BasicKits._BasicFuncT.find_file_encoding(html_filepath) is None else 'utf-8'
    # Open the HTML file and read it into a string
    with open(html_filepath, 'r', encoding=encodetype, errors=errors) as f:
        html_f = f.read()

    readability_cls = readability.Document(html_f, **kwargs)

    html_content = readability_cls.summary(html_partial)

    # Parse the HTML with BeautifulSoup
    soup = bs4.BeautifulSoup(html_content, 'html.parser')

    # Find and remove all strikethrough text
    try:
        for strike_tag in soup(strike_tags):
            strike_tag.decompose()
    except:
        warnings.warn('decompose of bs4 runs in error, automatically passed', ResourceWarning)

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
    if not suppress_warn:
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


def convert_doc_to_single_line_str(doc_file_path, suppress_warn=False):
    """
    new version use Libreoffice->soffice to read the txt from doc file.
    Args:
        doc_file_path: read doc file path, need antiword engine

    Returns: flat string with no \s{2,}, no \n, \t etc include, but only \s

    """

    if not suppress_warn:
        __sep_letter_warning()
    doc_file = _BasicKits._BasicFuncT.get_absolute_posix_path(doc_file_path)

    # get pure name
    doc_pure_name = os.path.splitext(os.path.basename(doc_file))[0]

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_text_file = os.path.join(temp_dir, f"{doc_pure_name}.txt")

        # Use LibreOffice's soffice command to convert the DOC file to a text file
        subprocess.run(["soffice", "--headless", "--convert-to", "txt:Text", "--outdir", temp_dir, doc_file])

        doc_encoding = _BasicKits._BasicFuncT.find_file_encoding(temp_text_file) \
            if _BasicKits._BasicFuncT.find_file_encoding(temp_text_file) else 'utf-8'

        # Read the converted text file， however, when you successful read, you always convert success.
        with open(temp_text_file, "r", encoding=doc_encoding) as text_file:
            result_text = text_file.read()

        # Remove newline characters to make the text a single line
        result_text = re.sub(r'\s+', ' ', result_text, flags=re.IGNORECASE)

        if len(result_text) == 0:
            warnings.warn('Errors occurs for read Nothing from Libre office converted file', UserWarning)

    return result_text


def convert_rtf_to_single_line_str(rtf_file_path, suppress_warn=False):
    """
    Args:
        rtf_file_path: the rtf file path

    Returns: flat string with no \s{2,}, no \n, \t etc include, but only \s

    """
    if not suppress_warn:
        __sep_letter_warning()
    # Convert the RTF file to TXT format
    result_text = pypandoc.convert_file(rtf_file_path, 'plain', format='rtf')

    # Remove newline characters to make the text a single line
    result_text = re.sub(r'\s+', ' ', result_text, flags=re.IGNORECASE)

    return result_text


# --------------------------------------------------------------------------
# Pdf part-slicer
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
