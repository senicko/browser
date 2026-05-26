import logging
import sys
from typing import cast

import joblib
import nltk
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import Normalizer

from model.bm25 import BM25Vectorizer

from .paths import ARTIFACTS_DIR
from .scraper import load_corpus, build_corpus
from .tokenizer import StemTokenizer


def main():
    nltk.download("punkt")
    nltk.download("punkt_tab")
    nltk.download("stopwords")

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    corpus = load_corpus()
    if corpus is None:
        corpus = build_corpus(logger)

    # Prepare BoW (Bag of Words) model

    corpus_texts = [document.text for document in corpus]

    tfidf_vectorizer = TfidfVectorizer(
        tokenizer=StemTokenizer(),
        min_df=5,
        max_df=0.8,
    )

    bow = tfidf_vectorizer.fit_transform(corpus_texts)
    logger.debug(f"vocabulary_size: {len(tfidf_vectorizer.vocabulary_)}")

    joblib.dump(bow, ARTIFACTS_DIR / "bow.pkl")
    joblib.dump(tfidf_vectorizer, ARTIFACTS_DIR / "vectorizer.pkl")

    # Prepare LSA (Latent Semantic Analysis) model

    n_components = min(
        200,
        cast(csr_matrix, bow).shape[0] - 1,
        cast(csr_matrix, bow).shape[1] - 1,
    )

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    normalizer = Normalizer(copy=False)
    lsa_pipeline = make_pipeline(svd, normalizer)

    lsa_bow = lsa_pipeline.fit_transform(bow)
    lsa_vectorizer = Pipeline([("tfidf", tfidf_vectorizer), ("svd", lsa_pipeline)])

    joblib.dump(lsa_bow, ARTIFACTS_DIR / "lsa_bow.pkl")
    joblib.dump(lsa_vectorizer, ARTIFACTS_DIR / "lsa_vectorizer.pkl")

    # Prepare BM25 model

    bm25_vectorizer = BM25Vectorizer(
        k1=2.0,  # Require more words to saturate
        b=0.4,  # Make sure to not unfailry boost too short articles
        delta=1,  # Use BM25+
        tokenizer=StemTokenizer(),
        min_df=5,
        max_df=0.8,
        norm=None,
    )

    bm25 = bm25_vectorizer.fit_transform(corpus_texts)
    joblib.dump(bm25, ARTIFACTS_DIR / "bm25.pkl")
    joblib.dump(bm25_vectorizer, ARTIFACTS_DIR / "bm25_vectorizer.pkl")

    loaded_bm25 = joblib.load(ARTIFACTS_DIR / "bm25.pkl")
    loaded_bm25_vectorizer = joblib.load(ARTIFACTS_DIR / "bm25_vectorizer.pkl")
    sample_bm25 = loaded_bm25_vectorizer.transform([corpus_texts[0]])
    if sample_bm25.shape[1] != loaded_bm25.shape[1]:
        raise RuntimeError(
            "Saved BM25 vectorizer shape does not match saved BM25 matrix"
        )


if __name__ == "__main__":
    main()
