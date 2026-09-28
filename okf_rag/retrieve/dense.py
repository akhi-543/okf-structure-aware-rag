import numpy as np


class DenseRetriever:
    """R0 baseline: dense cosine search over pre-computed chunk vectors.

    `vectors` rows are assumed L2-normalized (unit length) on the way in --
    scoring is a plain dot product, which equals cosine similarity only
    under that assumption. Callers must normalize before constructing this
    retriever (e.g. `model.encode(..., normalize_embeddings=True)`); an
    un-normalized vector silently turns the dot product into something
    other than cosine similarity, with no error raised.
    """

    def __init__(self, encode, vectors: np.ndarray, doc_ids: list[int], chunk_ids: list[int]):
        # (I9) Constructor order matches search()'s return shape
        # (doc_id, chunk_id, score) -- it used to be (chunk_ids, doc_ids),
        # inverted relative to that. doc_ids and chunk_ids are both
        # same-length, same-type int lists, so a caller transposing them at
        # a call site was silent and produced confidently wrong pairings
        # forever. The length check below at least catches a caller who
        # passes mismatched-length lists (a different bug, but one the old
        # code also didn't guard).
        if not (len(doc_ids) == len(chunk_ids) == len(vectors)):
            raise ValueError(
                f"DenseRetriever: doc_ids ({len(doc_ids)}), chunk_ids ({len(chunk_ids)}), "
                f"and vectors ({len(vectors)}) must all have the same length"
            )
        self.encode = encode
        self.vectors = vectors.astype(np.float32)   # rows assumed L2-normalized
        self.doc_ids = doc_ids
        self.chunk_ids = chunk_ids

    def search(self, query: str, k: int) -> list[tuple[int, int, float]]:
        q = self.encode([query])[0]
        scores = self.vectors @ q
        top = np.argsort(-scores)[:k]
        return [(self.doc_ids[i], self.chunk_ids[i], float(scores[i])) for i in top]
