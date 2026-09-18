"""Sequential collision assignment for semantic IDs.

Items sharing a full quantized prefix receive a final code in input iteration
order. The mapping passed to ``fit_transform`` is never modified.
"""

import logging
from abc import ABC, abstractmethod
from collections import defaultdict


class BaseCollisionSolver(ABC):
    """Group shared prefixes and append a collision code to each item."""

    name: str = "base"

    def __init__(self, codebook_size: int, extend_codebook: str = "global"):
        self.codebook_size = int(codebook_size)
        self.extend_codebook = extend_codebook

    @abstractmethod
    def _assign_group(self, colliding_items):
        """Return the ordered items and their collision codes."""

    def fit(self, item_codes):
        prefix2items = defaultdict(list)
        for item_id, codes in item_codes.items():
            prefix2items[tuple(codes)].append(item_id)

        self.max_collision_fact_ = max(
            len(colliding_items) for colliding_items in prefix2items.values()
        )
        logging.info(f"Max collisions for a semantic code: {self.max_collision_fact_}")

        self.effective_codebook_size_ = self.codebook_size
        self._extend_collision_codebook()
        self.prefix2items_ = prefix2items
        return self

    def _extend_collision_codebook(self):
        if self.max_collision_fact_ <= self.effective_codebook_size_:
            return
        if self.extend_codebook == "global":
            logging.warning(
                f"Collision codebook size increased from {self.effective_codebook_size_} "
                f"to {self.max_collision_fact_} to handle all collisions"
            )
            self.effective_codebook_size_ = self.max_collision_fact_
        elif self.extend_codebook == "per_prefix":
            logging.warning(
                f"Some prefixes exceed codebook_size={self.effective_codebook_size_} "
                f"(max collisions = {self.max_collision_fact_}); "
                f"only those prefixes will use an extended local range"
            )
        else:
            raise ValueError(f"Unknown extend_codebook: {self.extend_codebook!r}")

    def transform(self, item_codes):
        if not hasattr(self, "prefix2items_"):
            raise RuntimeError("call fit() (or fit_transform()) before transform()")
        out = {item_id: list(codes) for item_id, codes in item_codes.items()}
        logging.info(
            f"Solving collisions with solver={self.name}, "
            f"extend_codebook={self.extend_codebook}..."
        )
        for colliding_items in self.prefix2items_.values():
            ordered_items, collision_codes = self._assign_group(colliding_items)
            for item_id, code in zip(ordered_items, collision_codes):
                out[item_id].append(code)
        return out

    def fit_transform(self, item_codes):
        return self.fit(item_codes).transform(item_codes)


class SequentialSolver(BaseCollisionSolver):
    """Assign codes 0, 1, 2, ... in iteration order."""

    name = "sequential"

    def _assign_group(self, colliding_items):
        return colliding_items, list(range(len(colliding_items)))


def get_collision_solver(name: str, codebook_size: int, extend_codebook: str = "global"):
    """Construct the retained Sequential solver; reject unsupported names."""
    if name != "sequential":
        raise ValueError(f"Unknown solver: {name!r}")
    return SequentialSolver(codebook_size=codebook_size, extend_codebook=extend_codebook)
