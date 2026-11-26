"""GDSD, Xiaohang Tang et al., arXiv 2605.29398; code https://github.com/GaryBall/GDSD;
see package docstring for details."""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True, slots=True)
class GDSDRecord:
    """One iterative-remasking completion sampled from the behaviour policy."""

    request_id: int
    prompt_ids: list[int]
    completion_ids: list[int]
    reward: float | None = None
    group_id: int = 0
    version: int = 0

    @property
    def length(self) -> int:
        return len(self.completion_ids)

    def validate(self) -> None:
        if not self.prompt_ids or not self.completion_ids:
            raise ValueError("GDSD record needs a prompt and a completion")
        if self.version < 0:
            raise ValueError("GDSD record version cannot be negative")


@dataclasses.dataclass(frozen=True, slots=True)
class GDSDGenerationResult:
    sequences: list[list[int]]
    records: tuple[GDSDRecord, ...]
