from typing import List
from dataclasses import dataclass

@dataclass
class IOEvent:
    """A class to represent an IO event."""

    @classmethod
    def from_trace(cls, items: List[str]):
        return cls(*[int(item) for item in items])
