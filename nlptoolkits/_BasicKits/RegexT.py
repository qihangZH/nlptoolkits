import re
import numpy as np
import typing


def rolling_window_paragraph_selector(
        sentences: typing.List[str],
        window_size: int,
        regex_pattern: typing.Union[typing.List[str], str],
        combine_overlap_windows: bool = True,
        compile_flags: int = re.IGNORECASE
):
    """
    Read a Document contains multiple sentences,
    Then, use several regex patterns to check whether a sentence follow each pattern, for example,
    if there are M patterns, like [r'pattern1', r'pattern2', r'pattern3'], then for all sentences N.
    We finally got a numpy matrix of shape (N, M),
    where each element is 1 or 0, indicating whether the sentence follow the pattern.

    Then, the rolling window size means that how many consecutive sentences in this small window.
    We check whether the patterns provided in regex in that small window, each have at least one sentence to match.
    This do not require that all sentences in the small window should match the regex pattern.
    For example, if we have a regex pattern [r'pattern1', r'pattern2', r'pattern3'], and the sentences are
    ['sentence1', 'sentence2', 'sentence3', 'sentence4', 'sentence5'], then if sentence 1 match pattern2, sentence 2
    match pattern3, sentence 3 match pattern1, then we can say that this small window is valid.

    If this small window is valid, we mark this as a valid paragraph and mark the index of sentences in this window.
    Then, we will move the window to the next position, and check again.

    If combine overlap_windows is True, then for consecutive windows that have overlap, we will combine them into one.
    This is set to True default and finally for all paragraphs selected, there should no any index overlap between any
    two paragraphs.

    Args:
        sentences:
            A list of sentences. Each sentence is a string.
        window_size:
            The size of the rolling window. It should be a positive integer.
        regex_pattern:
            A regex pattern or a list of regex patterns to check against the sentences.
            If it is a list, each pattern will be checked separately.
            If it is a string, it will be treated as a single pattern.
        combine_overlap_windows:
            If True, then we will combine the overlapping windows,
            otherwise, we will just return the result of each window.
        compile_flags:
            The flags to compile the regex patterns. Default is re.IGNORECASE.
            See re.compile() for more details.

    Returns:

    """
    # ---------- argument checks ----------
    if window_size <= 0:
        raise ValueError("`window_size` must be a positive integer.")
    if not sentences or window_size > len(sentences):
        raise ValueError("`sentences` must contain at least `window_size` sentences.")

    patterns = [regex_pattern] if isinstance(regex_pattern, str) else list(regex_pattern)
    if not patterns:
        raise ValueError("`regex_pattern` must contain at least one pattern.")
    compiled = [re.compile(p, compile_flags) for p in patterns]

    # ---------- build match matrix ----------
    n_sent, n_pat = len(sentences), len(compiled)
    match_matrix = np.zeros((n_sent, n_pat), dtype=bool)
    for i, sent in enumerate(sentences):
        for j, pat in enumerate(compiled):
            if pat.search(sent):
                match_matrix[i, j] = True

    # ---------- scan windows ----------
    valid_windows: typing.List[typing.Tuple[int, int]] = []
    for start in range(n_sent - window_size + 1):
        end = start + window_size                      # exclusive bound
        if match_matrix[start:end].any(axis=0).all():  # every pat appears
            valid_windows.append((start, end))

    # ---------- optional merge ----------
    if not combine_overlap_windows:
        return valid_windows

    merged: typing.List[typing.Tuple[int, int]] = []
    for cur_start, cur_end in valid_windows:
        if not merged:
            merged.append((cur_start, cur_end))
            continue

        prev_start, prev_end = merged[-1]
        if cur_start < prev_end:                       # windows overlap
            merged[-1] = (prev_start, max(prev_end, cur_end))
        else:
            merged.append((cur_start, cur_end))

    return merged