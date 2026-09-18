from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class TokenTrieNode:
    children: dict[int, TokenTrieNode] = field(default_factory=dict)


class TokenTrie:
    def __init__(self) -> None:
        self.root = TokenTrieNode()

    def add(self, tokens: Iterable[int]) -> None:
        node = self.root

        for token in tokens:
            if token not in node.children:
                node.children[token] = TokenTrieNode()

            node = node.children[token]

    def get_allowed_next_tokens(self, prefix: Iterable[int]) -> list[int]:
        node = self.root

        for token in prefix:
            if token not in node.children:
                return []

            node = node.children[token]

        return list(node.children.keys())
