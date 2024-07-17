
library_remove_bigram_combinations = [
    # Pronoun-Pronoun (PRP-PRP, PRP$-PRP$, PRP-PRP$, PRP$-PRP)
    ('PRP', 'PRP'), ('PRP$', 'PRP$'), ('PRP', 'PRP$'), ('PRP$', 'PRP'),

    # Preposition-Preposition (IN-IN)
    ('IN', 'IN'),

    # Adverb-Adverb (RB-RB, RBR-RBR, RBS-RBS, RB-RBR, RB-RBS, RBR-RBS)
    ('RB', 'RB'), ('RBR', 'RBR'), ('RBS', 'RBS'), ('RB', 'RBR'), ('RB', 'RBS'), ('RBR', 'RBS'),

    # WH-Adverb-Adverb (WRB-RB, WRB-RBR, WRB-RBS)
    ('WRB', 'RB'), ('WRB', 'RBR'), ('WRB', 'RBS'),

    # Preposition-Adverb (IN-RB, IN-RBR, IN-RBS)
    ('IN', 'RB'), ('IN', 'RBR'), ('IN', 'RBS'),

    # Adverb-Preposition (RB-IN, RBR-IN, RBS-IN)
    ('RB', 'IN'), ('RBR', 'IN'), ('RBS', 'IN'),

    # Preposition-WH-Adverb (IN-WRB)
    ('IN', 'WRB'),

    # WH-Adverb-Preposition (WRB-IN)
    ('WRB', 'IN'),

    # Preposition-Determiner (IN-DT)
    ('IN', 'DT'),

    # Determiner-Preposition (DT-IN)
    ('DT', 'IN'),

    # Adverb-WH-Adverb (RB-WRB, RBR-WRB, RBS-WRB)
    ('RB', 'WRB'), ('RBR', 'WRB'), ('RBS', 'WRB'),

    # WH-Adverb-Adverb (WRB-RB, WRB-RBR, WRB-RBS)
    ('WRB', 'RB'), ('WRB', 'RBR'), ('WRB', 'RBS'),

    # Adverb-Determiner (RB-DT, RBR-DT, RBS-DT)
    ('RB', 'DT'), ('RBR', 'DT'), ('RBS', 'DT'),

    # Determiner-Adverb (DT-RB, DT-RBR, DT-RBS)
    ('DT', 'RB'), ('DT', 'RBR'), ('DT', 'RBS'),

    # WH-Adverb-Determiner (WRB-DT)
    ('WRB', 'DT'),

    # Determiner-WH-Adverb (DT-WRB)
    ('DT', 'WRB')
]

library_remove_bigram_combinations_lower = [
            tuple([s.lower() for s in tp])
            for tp in library_remove_bigram_combinations
        ]

library_remove_single_words_lower = ["i", "ive", "youve", "weve", "im", "youre", "were", "id", "youd", "wed", "thats",
                                     "that"]
