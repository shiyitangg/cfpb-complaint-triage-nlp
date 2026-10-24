"""Model specifications. All are logistic regression on sparse features fitted on train only."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder

STRUCTURED = ["product", "sub_product", "issue", "sub_issue"]


@dataclass(frozen=True)
class Spec:
    name: str
    text: bool
    categorical: tuple[str, ...]
    text_col: str = "text"

    @property
    def columns(self) -> list[str]:
        return ([self.text_col] if self.text else []) + list(self.categorical)


SPECS = {
    "structured": Spec("structured", False, tuple(STRUCTURED)),
    "tfidf": Spec("tfidf", True, ()),
    "tfidf_structured": Spec("tfidf_structured", True, tuple(STRUCTURED)),  # main model
    "tfidf_structured_company": Spec("tfidf_structured_company", True, tuple(STRUCTURED) + ("company",)),  # ablation
    # Ablation: company names inside the narrative replaced by one placeholder (see company_mask.py).
    "tfidf_structured_masked": Spec("tfidf_structured_masked", True, tuple(STRUCTURED), text_col="text_masked"),
}


class Featurizer:
    """TF-IDF on text plus one-hot on categorical columns. Fit on training rows only.

    The TF-IDF vocabulary and idf weights are fitted on a random sample of training rows
    (vocab_sample) to bound memory; all rows are then transformed with that vocabulary.
    """

    def __init__(self, spec: Spec, min_df: int = 5, max_features: int = 200_000, cat_min_frequency: int = 20,
                 vocab_sample: int | None = 200_000, seed: int = 0, tfidf: TfidfVectorizer | None = None):
        """Pass an already fitted `tfidf` to share one vocabulary across models; it is then not refitted."""
        self.spec = spec
        self.vocab_sample = vocab_sample
        self.seed = seed
        self._tfidf_prefitted = tfidf is not None
        self.tfidf = tfidf if tfidf is not None else TfidfVectorizer(ngram_range=(1, 2), min_df=min_df, max_features=max_features,
                                     sublinear_tf=True, lowercase=True, dtype=np.float32) if spec.text else None
        self.ohe = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=cat_min_frequency,
                                 dtype=np.float32) if spec.categorical else None

    def _cat(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[list(self.spec.categorical)].astype("object").fillna("<none>")

    def fit(self, train: pd.DataFrame) -> "Featurizer":
        if self.tfidf is not None and not self._tfidf_prefitted:
            t = train[self.spec.text_col]
            if self.vocab_sample and len(t) > self.vocab_sample:
                t = t.sample(self.vocab_sample, random_state=self.seed)
            self.tfidf.fit(t)
            # Pruned terms are kept only for introspection and can be millions of bigrams; drop before saving.
            self.tfidf.stop_words_ = None
        if self.ohe is not None:
            self.ohe.fit(self._cat(train))
        return self

    def transform(self, df: pd.DataFrame, text_matrix: sp.csr_matrix | None = None) -> sp.csr_matrix:
        """text_matrix: precomputed tfidf.transform(df[text_col]), to avoid transforming the same rows twice."""
        parts = []
        if self.tfidf is not None:
            parts.append(text_matrix if text_matrix is not None else self.tfidf.transform(df[self.spec.text_col]))
        if self.ohe is not None:
            parts.append(sp.csr_matrix(self.ohe.transform(self._cat(df))))
        return sp.hstack(parts, format="csr")

    def feature_names(self) -> np.ndarray:
        names = []
        if self.tfidf is not None:
            names.append(np.array(["text:" + w for w in self.tfidf.get_feature_names_out()]))
        if self.ohe is not None:
            names.append(self.ohe.get_feature_names_out(list(self.spec.categorical)))
        return np.concatenate(names)
