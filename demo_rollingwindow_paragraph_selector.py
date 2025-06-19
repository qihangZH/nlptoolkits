from nlptoolkits._BasicKits.RegexT import rolling_window_paragraph_selector
# --------------------------------------------
# Synthetic document with 12 short sentences
# --------------------------------------------
doc_sentences = [
    "Because the sensors were recalibrated, measurement drift was eliminated.",   # 0
    "The field team collected new samples every six hours.",                      # 1
    "However, a sudden storm halted the expedition for two days.",                # 2
    "Therefore, the schedule slipped by roughly 18 hours.",                       # 3
    "In conclusion, environmental variance was the major source of delay.",       # 4
    # –– new paragraph ––
    "Because the API changed, the client libraries had to be refactored.",        # 5
    "Developers introduced integration tests to catch regressions early.",        # 6
    "However, the staging database lacked anonymised data.",                      # 7
    "Therefore, the rollout paused until compliance approved the dataset.",       # 8
    "In conclusion, proactive testing accelerated the second deployment.",        # 9
    # –– final paragraph ––
    "Because quarterly demand surged, production lines ran 24/7.",                # 10
    "Therefore, maintenance windows were compressed to 90 minutes."               # 11
]

# -----------------------------------------------------
# Regex patterns we want to see PRESENT in each window
# -----------------------------------------------------
regexes = [r"because", r"however", r"therefore", r"conclusion"]

# -----------------
# Pick window size
# -----------------
window = 4     # → look at 6 consecutive sentences at a time

# -----------------------------------------------------------------
# Call the function (assumes it’s already defined in your session)
# -----------------------------------------------------------------
paragraph_ranges = rolling_window_paragraph_selector(
    sentences=doc_sentences,
    window_size=window,
    regex_pattern=regexes,
    combine_overlap_windows=True   # try False as well to see raw windows
)

print("Returned ranges (start, end_exclusive):", paragraph_ranges)

# -------------------------------------------------------
# Show each extracted paragraph for easy visual checking
# -------------------------------------------------------
for start, end in paragraph_ranges:
    print(f"\nParagraph covering sentences[{start}:{end}]:")
    for i, sent in enumerate(doc_sentences[start:end], start=start):
        print(f"{i:>2}: {sent}")
