from nlptoolkits.SmallKits import IOHandlerT

if __name__ == '__main__':
    texture = IOHandlerT._ocr_pdf_to_text(
        './input_data/ocr_sample.pdf',
        start_index=0,
        end_index=1
    )
    # only first page get

    print(texture)

    texture = IOHandlerT.convert_ocrpdf_to_single_line_str(
        './input_data/ocr_sample.pdf',
        start_index=0,
        end_index=1
    )
    # only first page get

    print(texture)