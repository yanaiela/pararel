from dataclasses import dataclass


@dataclass(frozen=True)
class PatternNode:
    lm_pattern: str
    spike_pattern: str
    lemma: str
    extended_lemma: str
    tense: str
    example: str = None

    wiki_occurence: int = None

    def __str__(self):
        return self.lm_pattern


@dataclass(frozen=True)
class EdgeType:
    syntactic_change: str
    lexical_change: str
    determiner_change: str

