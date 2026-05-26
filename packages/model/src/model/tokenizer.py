import re
from typing import Callable
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

class StemTokenizer:
    def __init__(
        self,
        stemmer: Callable[[str], str] = PorterStemmer().stem,
        tokenizer: Callable[[str], list[str]] = word_tokenize,
        stopwords: set[str] = set(stopwords.words("english")),
    ):
        self.stemmer = stemmer
        self.tokenizer = tokenizer
        self.stopwords = stopwords

    def token_count(self, text: str) -> int:
        """Returns number of tokens in the passed text."""
        return len(self(text))

    def __call__(self, text: str) -> list[str]:
        clean_txt = re.sub(r"\n", " ", text)
        clean_txt = re.sub(r"\s+", " ", clean_txt)
        clean_txt = clean_txt.strip()

        tokens = self.tokenizer(clean_txt)

        return [
            self.stemmer(token.lower())
            for token in tokens
            if token.isalpha() and token.lower() not in self.stopwords
        ]
