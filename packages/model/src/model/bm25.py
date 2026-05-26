# Too much library specific code to handle strict typing
# pyright: reportCallIssue=false, reportAttributeAccessIssue=false, reportArgumentType=false,  reportIncompatibleMethodOverride=false

import warnings

import numpy as np
import scipy.sparse as sp
from sklearn.base import (
    BaseEstimator,
    OneToOneFeatureMixin,
    TransformerMixin,
    _fit_context,
)
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import CountVectorizer, _document_frequency
from sklearn.preprocessing import normalize
from sklearn.utils._param_validation import Interval, RealNotInt, StrOptions
from sklearn.utils.fixes import _IS_32BIT
from sklearn.utils.validation import FLOAT_DTYPES, check_is_fitted, validate_data


class BM25Transformer(
    OneToOneFeatureMixin, TransformerMixin, BaseEstimator, auto_wrap_output_keys=None
):
    _parameter_constraints: dict = {
        "k1": [Interval(RealNotInt, 0, None, closed="left")],
        "b": [Interval(RealNotInt, 0, 1, closed="both")],
        "norm": [StrOptions({"l1", "l2"}), None],
        "use_idf": ["boolean"],
        "smooth_idf": ["boolean"],
        "sublinear_tf": ["boolean"],
    }

    def __init__(
        self,
        *,
        k1=1.2,  # In absence of optimization 1.2 or 2.0
        b=0.75,  # In absence of optimization 0.75
        norm=None,
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=False,
        delta=0,  # This allows to upgrade to BM25+
    ):
        self.k1 = k1
        self.b = b
        self.norm = norm
        self.use_idf = use_idf
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf
        self.delta = delta

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """Learn BM25 vector

        Parameters
        ----------
        X : sparse_matrix of shape (n_samples, n_features)
            A matrix of term/token counts

        y : None
            This parameter is not needed to compute BM25

        Returns
        -------
        self : object
            Fitted transformer
        """

        X = validate_data(
            self,
            X,
            accept_sparse=("csr", "csc"),
            accept_large_sparse=not _IS_32BIT,
        )

        if not sp.issparse(X):
            X = sp.csr_matrix(X)

        dtype = X.dtype if X.dtype in (np.float64, np.float32) else np.float64

        # Find average document length (with respect to dictionary tokens)
        self.avgdl_ = X.sum(axis=1).mean()

        if self.use_idf:
            n_samples, _ = X.shape
            df = _document_frequency(X)
            df = df.astype(dtype, copy=False)

            df += float(self.smooth_idf)
            n_samples += int(self.smooth_idf)

            self.idf_ = np.log(n_samples / df) + 1

        return self

    def transform(self, X, copy=True):
        """Transform a count matrix to a BM25 representation.

        Parameters
        ----------
        X : sparse matrix of (n_samples, n_features)
            A matrix of term/token counts.

        copy : bool, default=True
            Whether to copy X and operate on the copy or perform in-place
            operations. `copy=False` will only be effective with CSR sparse matrix.

        Returns
        -------
        vectors : sparse matrix of shape (n_samples, n_features)
            BM25 matrix.
        """
        check_is_fitted(self)

        X = validate_data(
            self,
            X,
            accept_sparse="csr",
            dtype=[np.float64, np.float32],
            copy=copy,
            reset=False,
        )

        if not sp.issparse(X):
            X = sp.csr_matrix(X, dtype=X.dtype)

        dl = np.asarray(X.sum(axis=1)).ravel()  # Get length of each document

        # NOTE: Repeat length of each document number of times this document has non-zero
        #       frequency words, so that we can use np vectorization below.
        dl = np.repeat(dl, np.diff(X.indptr))

        if self.sublinear_tf:
            np.log(X.data, X.data)
            X.data += 1

        X.data = (X.data * (self.k1 + 1)) / (
            X.data + self.k1 * (1 - self.b + self.b * dl / self.avgdl_)
        ) + self.delta

        if hasattr(self, "idf_"):
            # the columns of X (CSR matrix) can be accessed with `X.indices `and
            # multiplied with the corresponding `idf` value
            X.data *= self.idf_[X.indices]

        if self.norm is not None:
            X = normalize(X, norm=self.norm, copy=False)

        return X


class BM25Vectorizer(CountVectorizer):
    _parameter_constraints: dict = {**CountVectorizer._parameter_constraints}
    _parameter_constraints.update(
        {
            "k1": [Interval(RealNotInt, 0, None, closed="left")],
            "b": [Interval(RealNotInt, 0, 1, closed="both")],
            "norm": [StrOptions({"l1", "l2"}), None],
            "use_idf": ["boolean"],
            "smooth_idf": ["boolean"],
            "sublinear_tf": ["boolean"],
        }
    )

    def __init__(
        self,
        *,
        input="content",
        encoding="utf-8",
        decode_error="strict",
        strip_accents=None,
        lowercase=True,
        preprocessor=None,
        tokenizer=None,
        analyzer="word",
        stop_words=None,
        token_pattern=r"(?u)\b\w\w+\b",
        ngram_range=(1, 1),
        max_df=1.0,
        min_df=1,
        max_features=None,
        vocabulary=None,
        binary=False,
        dtype=np.float32,
        k1=1.2,
        b=0.75,
        norm=None,
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=False,
        delta=0,
    ):
        super().__init__(
            input=input,
            encoding=encoding,
            decode_error=decode_error,
            strip_accents=strip_accents,
            lowercase=lowercase,
            preprocessor=preprocessor,
            tokenizer=tokenizer,
            analyzer=analyzer,
            stop_words=stop_words,
            token_pattern=token_pattern,
            ngram_range=ngram_range,
            max_df=max_df,
            min_df=min_df,
            max_features=max_features,
            vocabulary=vocabulary,
            binary=binary,
            dtype=dtype,
        )
        self.k1 = k1
        self.b = b
        self.norm = norm
        self.use_idf = use_idf
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf
        self.delta = delta

    @property
    def idf_(self):
        if not hasattr(self, "_bm25"):
            raise NotFittedError(
                f"{self.__class__.__name__} is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this attribute."
            )

        return self._bm25.idf_

    @idf_.setter
    def idf_(self, value):
        if not self.use_idf:
            raise ValueError("`idf_` cannot be set when `use_idf=False`.")

        if not hasattr(self, "_bm25"):
            self._bm25 = BM25Transformer(
                k1=self.k1,
                b=self.b,
                norm=self.norm,
                use_idf=self.use_idf,
                smooth_idf=self.smooth_idf,
                sublinear_tf=self.sublinear_tf,
                delta=self.delta,
            )

        self._validate_vocabulary()

        if hasattr(self, "vocabulary_"):
            if len(self.vocabulary_) != len(value):
                raise ValueError(
                    "idf length = %d must be equal to vocabulary size = %d"
                    % (len(value), len(self.vocabulary_))
                )

        self._bm25.idf_ = value

    def _check_params(self):
        if self.dtype not in FLOAT_DTYPES:
            warnings.warn(
                "Only {} 'dtype' should be used. {} 'dtype' will "
                "be converted to np.float64.".format(FLOAT_DTYPES, self.dtype),
                UserWarning,
            )

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, raw_documents, y=None):
        self._check_params()
        self._warn_for_unused_params()
        self._bm25 = BM25Transformer(
            k1=self.k1,
            b=self.b,
            norm=self.norm,
            use_idf=self.use_idf,
            smooth_idf=self.smooth_idf,
            sublinear_tf=self.sublinear_tf,
            delta=self.delta,
        )

        X = super().fit_transform(raw_documents)
        self._bm25.fit(X)

        return self

    def fit_transform(self, raw_documents, y=None):
        self._check_params()
        self._bm25 = BM25Transformer(
            k1=self.k1,
            b=self.b,
            norm=self.norm,
            use_idf=self.use_idf,
            smooth_idf=self.smooth_idf,
            sublinear_tf=self.sublinear_tf,
            delta=self.delta,
        )

        X = super().fit_transform(raw_documents)
        self._bm25.fit(X)

        return self._bm25.transform(X, copy=False)

    def transform(self, raw_documents):
        check_is_fitted(self, msg="The BM25 vectorizer is not fitted")

        X = super().transform(raw_documents)
        return self._bm25.transform(X, copy=False)
