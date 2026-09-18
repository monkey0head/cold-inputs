from abc import ABC, abstractmethod
from typing import Self

import numpy as np


class Quantizer(ABC):
    def __init__(self, num_codebooks: int, codebook_size: int) -> None:
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size

    @abstractmethod
    def fit(self, X: np.ndarray) -> Self: ...

    @abstractmethod
    def encode(self, X: np.ndarray) -> np.ndarray: ...

    def decode(self, codes: np.ndarray) -> np.ndarray | None:
        pass

    def reconstruct(self, X: np.ndarray) -> np.ndarray | None:
        """Encode X and decode back to embedding space.

        Args:
            X: float32 array of shape (N, D) — input embeddings.

        Returns:
            np.ndarray: float32 array of shape (N, D) — reconstructed embeddings.

        """
        reconstructed = self.decode(self.encode(X))

        if reconstructed is None:
            print(f"{self.__class__.__name__} does not implement decode()")

        return reconstructed

    def reconstruct_per_level(self, X: np.ndarray) -> np.ndarray | None:
        """Reconstruct X using only the first m codebooks for m = 1..num_codebooks.

        Only meaningful for hierarchical/residual quantizers where codebooks
        encode shared information about the whole vector. Returns None for
        quantizers where the notion of "first m levels" does not apply
        (e.g. product quantization, where codebooks describe disjoint
        subspaces).

        Args:
            X: float32 array of shape (N, D) — input embeddings.

        Returns:
            np.ndarray: float32 array of shape (L, N, D) — per-level
            reconstructions, where L = num_codebooks.

        """
        print(f"{self.__class__.__name__} does not implement reconstruct_per_level()")
        return None



    def fit_encode(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.encode(X)
