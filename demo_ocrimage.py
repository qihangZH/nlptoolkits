from nlptoolkits.SmallKits import IOHandlerT

if __name__ == '__main__':
    texture = IOHandlerT._ocr_image_to_text(
        './input_data/10.1257#0002828041301407.png'
    )
    # only first page get

    # print green color text
    print(f'\033[32m{texture}\033[0m')

    # also, plain text.
    # we can find that, this will loss some information, like paragraph. Thus we need to be more carefully.

    texture = IOHandlerT.convert_ocrimage_to_single_line_str(
        './input_data/10.1257#0002828041301407.png'
    )

    # print yellow color text
    print(f'\033[33m{texture}\033[0m')