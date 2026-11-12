"""Baselines the learned encoder has to beat.

All of them see exactly the same canonical notes as the encoder.

Retrieval
    ``jaccard``        overlap of the two note sets
    ``tfidf_notes``    TF-IDF over whole notes, cosine
    ``tfidf_words``    TF-IDF over notes *and* their words, cosine; this is
                       the strongest lexical baseline because "white musk"
                       and "musk" share a feature
    ``lsa``            128-d truncated SVD of ``tfidf_words``: the classic
                       way to get a dense "embedding" without a neural net

Label prediction
    one-vs-rest logistic regression on the ``tfidf_words`` features
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

from .features import Corpus, record_notes_with_positions
from .textnorm import note_tokens


def note_lists(corpus: Corpus) -> List[List[str]]:
    return [record_notes_with_positions(row)[0] for _, row in corpus.records.iterrows()]


class BagOfNotes:
    """Sparse count features; the vocabulary is fixed from the training split."""

    def __init__(self, train_notes: Sequence[Sequence[str]], with_words: bool) -> None:
        self.with_words = with_words
        self.column: Dict[str, int] = {}
        for notes in train_notes:
            for feature in self._features(notes):
                if feature not in self.column:
                    self.column[feature] = len(self.column)

    def _features(self, notes: Sequence[str]) -> List[str]:
        features = ["note:" + note for note in notes]
        if self.with_words:
            for note in notes:
                tokens = note_tokens(note)
                if len(tokens) > 1:
                    features.extend("word:" + token for token in tokens)
                else:
                    features.append("word:" + note)
        return features

    def transform(self, note_lists_: Sequence[Sequence[str]]) -> sparse.csr_matrix:
        rows, columns = [], []
        for row, notes in enumerate(note_lists_):
            for feature in set(self._features(notes)):
                column = self.column.get(feature)
                if column is not None:
                    rows.append(row)
                    columns.append(column)
        data = np.ones(len(rows), dtype=np.float32)
        return sparse.csr_matrix(
            (data, (rows, columns)), shape=(len(note_lists_), len(self.column))
        )


class BaselineSuite:
    """Fits every baseline on the training split and exposes record vectors."""

    def __init__(self, corpus: Corpus, lsa_dim: int = 128, seed: int = 7) -> None:
        self.corpus = corpus
        self.notes = note_lists(corpus)
        train_mask = (corpus.records["split"] == "train").to_numpy()
        train_notes = [notes for notes, keep in zip(self.notes, train_mask) if keep]

        self.binary = BagOfNotes(train_notes, with_words=False)
        self.binary_matrix = self.binary.transform(self.notes)

        self.tfidf_notes_model = TfidfTransformer(sublinear_tf=False)
        self.tfidf_notes_model.fit(self.binary_matrix[train_mask])
        self.tfidf_notes = normalize(self.tfidf_notes_model.transform(self.binary_matrix))

        self.words = BagOfNotes(train_notes, with_words=True)
        words_matrix = self.words.transform(self.notes)
        self.tfidf_words_model = TfidfTransformer(sublinear_tf=False)
        self.tfidf_words_model.fit(words_matrix[train_mask])
        self.tfidf_words = normalize(self.tfidf_words_model.transform(words_matrix))

        self.lsa_model = TruncatedSVD(n_components=lsa_dim, random_state=seed)
        self.lsa_model.fit(self.tfidf_words[train_mask])
        self.lsa = normalize(self.lsa_model.transform(self.tfidf_words)).astype(np.float32)

    def similarity(self, method: str, queries: np.ndarray, gallery: np.ndarray) -> np.ndarray:
        if method == "jaccard":
            query_matrix = self.binary_matrix[queries]
            gallery_matrix = self.binary_matrix[gallery]
            intersection = np.asarray((query_matrix @ gallery_matrix.T).todense())
            query_sizes = np.asarray(query_matrix.sum(axis=1))
            gallery_sizes = np.asarray(gallery_matrix.sum(axis=1)).T
            union = query_sizes + gallery_sizes - intersection
            return intersection / np.maximum(union, 1.0)
        if method == "tfidf_notes":
            return np.asarray((self.tfidf_notes[queries] @ self.tfidf_notes[gallery].T).todense())
        if method == "tfidf_words":
            return np.asarray((self.tfidf_words[queries] @ self.tfidf_words[gallery].T).todense())
        if method == "lsa":
            return self.lsa[queries] @ self.lsa[gallery].T
        raise ValueError(f"unknown baseline {method!r}")

    def fit_label_models(self, max_iter: int = 300) -> None:
        """Logistic regression from bag-of-notes to accords, family and subfamily."""
        corpus = self.corpus
        train_mask = (corpus.records["split"] == "train").to_numpy()
        features = self.tfidf_words

        accord_rows = np.flatnonzero(train_mask & corpus.has_accords)
        self.accord_models = []
        for accord in range(corpus.accord_targets.shape[1]):
            model = LogisticRegression(C=2.0, max_iter=max_iter)
            model.fit(features[accord_rows], corpus.accord_targets[accord_rows, accord] > 0)
            self.accord_models.append(model)

        family_rows = np.flatnonzero(train_mask & (corpus.family >= 0) & (corpus.subfamily >= 0))
        self.family_model = LogisticRegression(C=2.0, max_iter=max_iter)
        self.family_model.fit(features[family_rows], corpus.family[family_rows])
        self.subfamily_model = LogisticRegression(C=2.0, max_iter=max_iter)
        self.subfamily_model.fit(features[family_rows], corpus.subfamily[family_rows])

    def accord_scores(self, indices: np.ndarray) -> np.ndarray:
        features = self.tfidf_words[indices]
        return np.column_stack(
            [model.predict_proba(features)[:, 1] for model in self.accord_models]
        )

    def class_scores(self, kind: str, indices: np.ndarray, n_classes: int) -> np.ndarray:
        model = self.family_model if kind == "family" else self.subfamily_model
        probabilities = model.predict_proba(self.tfidf_words[indices])
        scores = np.zeros((len(indices), n_classes), dtype=np.float32)
        scores[:, model.classes_] = probabilities
        return scores
