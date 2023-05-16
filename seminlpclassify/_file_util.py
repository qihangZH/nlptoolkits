import functools
import signal
import socket
import time
from urllib.parse import urlparse


"""functools, the lowest level in structure"""


def line_counter(a_file):
    """Count the number of lines in a text file
    
    Arguments:
        a_file {str or Path} -- input_data text file
    
    Returns:
        int -- number of lines in the file
    """
    n_lines = 0
    with open(a_file, "rb") as f:
        n_lines = sum(1 for _ in f)
    return n_lines


def file_to_list(a_file):
    """Read a text file to a list, each line is an element
    
    Arguments:
        a_file {str or path} -- path to the file
    
    Returns:
        [str] -- list of lines in the input_data file, can be empty
    """
    file_content = []
    with open(a_file, "rb") as f:
        for l in f:
            file_content.append(l.decode(encoding="utf-8").strip())
    return file_content


def list_to_file(list, a_file, validate=True):
    """Write a list to a file, each element in a line
    The strings needs to have no line break "\n" or they will be removed
    
    Keyword Arguments:
        validate {bool} -- check if number of lines in the file
            equals to the length of the list (default: {True})
    """
    with open(a_file, "w", 8192000, encoding="utf-8", newline="\n") as f:
        for e in list:
            e = str(e).replace("\n", " ").replace("\r", " ")
            f.write("{}\n".format(e))
    if validate:
        assert line_counter(a_file) == len(list)


def read_large_file(a_file, block_size=10000):
    """A generator to read text files into blocks
    Usage: 
    for block in read_large_file(filename):
        do_something(block)
    
    Arguments:
        a_file {str or path} -- path to the file
    
    Keyword Arguments:
        block_size {int} -- [number of lines in a block] (default: {10000})
    """
    block = []
    with open(a_file) as file_handler:
        for line in file_handler:
            block.append(line)
            if len(block) == block_size:
                yield block
                block = []
    # yield the last block
    if block:
        yield block


def check_server(url):
    parsed_url = urlparse(url)
    host = parsed_url.hostname
    port = parsed_url.port
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5)  # You can adjust timeout as needed
    try:
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except socket.gaierror:
        return False


def threads_interrupt_initiator():
    """
    Function for multiprocessing.Pool(initializer=threads_interrupt_initiator())
    Each pool process will execute this as part of its initialization.
    Use this to keep safe for multiprocessing...and gracefully interrupt by keyboard
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def timer_wrapper(func):
    """Wrapper for function running timing"""

    @functools.wraps(func)
    def decorated(*args, **kwargs):
        time_start = time.time()
        result = func(*args, **kwargs)
        time_end = time.time()
        time_spend = time_end - time_start
        print('%s cost time: %.5f s' % (func.__name__, time_spend))
        return result

    return decorated


