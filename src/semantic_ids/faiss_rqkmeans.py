from typing import Self

import faiss
import numpy as np
from faiss import ResidualQuantizer

from src.semantic_ids._quantizer import Quantizer


class FaissRQKMeans(Quantizer):
    """Fast K-means based residual quantizer.

    Adapted from https://github.com/AkaliKong/MiniOneRec/blob/main/rq/rqkmeans_faiss.py
    """

    def __init__(self, num_codebooks: int, codebook_size: int) -> None:
        super().__init__(num_codebooks, codebook_size)

        self._nbits = int(np.log2(self.codebook_size))

    def fit(self, X: np.ndarray) -> Self:
        d = X.shape[1]

        faiss_rq = ResidualQuantizer(d, self.num_codebooks, self._nbits)
        faiss_rq.train_type = ResidualQuantizer.Train_default
        faiss_rq.max_beam_size = 1

        faiss_rq.train(np.ascontiguousarray(X.astype(np.float32)))

        self._faiss_rq = faiss_rq
        self._d = d

        return self

    def encode(self, X: np.ndarray) -> np.ndarray:

        X = np.ascontiguousarray(X.astype(np.float32))

        codes_packed = self._faiss_rq.compute_codes(X)

        if self._nbits % 8 == 0:
            codes = codes_packed.astype(np.int32)
        else:
            codes = self._unpack_codes(codes_packed)

        if codes_packed.ndim == 1:
            n_bytes = (self._faiss_rq.M * self._nbits + 7) // 8
            codes_packed = codes_packed.reshape(-1, n_bytes)

        codes = codes.astype(np.int32)

        return codes

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """Encode X with the FAISS residual quantizer and decode back to embedding space.

        Args:
            X: float32 array of shape (N, D) — input embeddings.

        Returns:
            np.ndarray: float32 array of shape (N, D) — reconstructed embeddings
            obtained by encoding to packed codes and decoding via the FAISS index.
        """
        X = np.ascontiguousarray(X.astype(np.float32))
        codes_packed = self._faiss_rq.compute_codes(X)
        return self._faiss_rq.decode(codes_packed)

    def reconstruct_per_level(self, X: np.ndarray) -> np.ndarray:

        codes = self.encode(X)
        d = X.shape[1]
        codebooks = faiss.vector_to_array(self._faiss_rq.codebooks).reshape(
            self.num_codebooks, self.codebook_size, d
        )

        rec = np.zeros((X.shape[0], d), dtype=np.float32)
        per_level = []
        for m in range(self.num_codebooks):
            rec = rec + codebooks[m][codes[:, m]]
            per_level.append(rec.copy())
        return np.stack(per_level, axis=0)

    def _unpack_codes(self, codes: np.ndarray) -> np.ndarray:
        """
        Unpack FAISS's bit-packed codes into integer index arrays

        Args:
            codes (np.ndarray): uint8 array of shape (N, M_bytes)

        Returns:
            np.ndarray: int32 array of shape (N, num_codebooks) containing unpacked indices
        """
        N = codes.shape[0]

        # FAISS uses Little Endian packing
        packed_ints = np.zeros(N, dtype=np.int64)

        for i in range(codes.shape[1]):
            packed_ints |= codes[:, i].astype(np.int64) << (8 * i)

        unpacked_codes = np.zeros((N, self.num_codebooks), dtype=np.int32)
        mask = (1 << self._nbits) - 1  # e.g., mask for 9 bits is 511 (0x1FF)

        for i in range(self.num_codebooks):
            unpacked_codes[:, i] = (packed_ints >> (i * self._nbits)) & mask

        return unpacked_codes
