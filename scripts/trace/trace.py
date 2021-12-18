from abc import abstractmethod
from pathlib import Path

class Trace:
    @abstractmethod
    def simulate(self, path: Path):
        pass
