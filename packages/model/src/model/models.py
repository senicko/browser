from functools import lru_cache
from typing import Final, Protocol, cast

import joblib
from pydantic import BaseModel
from schema import Document
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .paths import ARTIFACTS_DIR


@lru_cache
def load_corpus() -> list[Document]:
    return joblib.load(ARTIFACTS_DIR / "corpus.pkl")


@lru_cache
def load_lsa_bow():
    return joblib.load(ARTIFACTS_DIR / "lsa_bow.pkl")


@lru_cache
def load_lsa_vectorizer():
    return joblib.load(ARTIFACTS_DIR / "lsa_vectorizer.pkl")


@lru_cache
def load_bow():
    return joblib.load(ARTIFACTS_DIR / "bow.pkl")


@lru_cache
def load_vectorizer():
    return joblib.load(ARTIFACTS_DIR / "vectorizer.pkl")


@lru_cache
def load_bm25():
    return joblib.load(ARTIFACTS_DIR / "bm25.pkl")


@lru_cache
def load_bm25_vectorizer():
    return joblib.load(ARTIFACTS_DIR / "bm25_vectorizer.pkl")


class Model(Protocol):
    def query(self, phrase: str, top_n: int = 10) -> list["ModelResponse"]: ...


class ModelResponse(BaseModel):
    title: str
    url: str
    score: float
    text: str


class BoWModel:
    @staticmethod
    def query(phrase: str, top_n: int = 10) -> list[ModelResponse]:
        corpus: Final[list[Document]] = load_corpus()
        phrase_vec = load_vectorizer().transform([phrase])
        similarities = cosine_similarity(phrase_vec, load_bow()).flatten()
        scores = similarities.argsort()[-top_n:][::-1]

        return [
            ModelResponse(
                title=corpus[i].title,
                text=corpus[i].text,
                url=corpus[i].url,
                score=cast(float, similarities[i]),
            )
            for i in scores
        ]


class LSAModel:
    @staticmethod
    def query(phrase: str, top_n: int = 10) -> list[ModelResponse]:
        corpus: Final[list[Document]] = load_corpus()
        phrase_vec = load_lsa_vectorizer().transform([phrase])
        similarities = cosine_similarity(phrase_vec, load_lsa_bow()).flatten()
        scores = similarities.argsort()[-top_n:][::-1]

        return [
            ModelResponse(
                title=corpus[i].title,
                text=corpus[i].text,
                url=corpus[i].url,
                score=cast(float, similarities[i]),
            )
            for i in scores
        ]


class BM25Model:
    @staticmethod
    def query(phrase: str, top_n: int = 10) -> list[ModelResponse]:
        corpus: Final[list[Document]] = load_corpus()
        bm25_vectorizer = load_bm25_vectorizer()
        phrase_vec = CountVectorizer.transform(bm25_vectorizer, [phrase])

        if phrase_vec.nnz == 0:
            return []

        phrase_vec.data.fill(1)
        scores = (load_bm25() @ phrase_vec.T).toarray().ravel()
        top_scores = scores.argsort()[-top_n:][::-1]

        return [
            ModelResponse(
                title=corpus[i].title,
                text=corpus[i].text,
                url=corpus[i].url,
                score=cast(float, scores[i]),
            )
            for i in top_scores
            if scores[i] > 0
        ]
