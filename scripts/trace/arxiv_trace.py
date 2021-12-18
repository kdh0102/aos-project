from  dataclasses import dataclass, field
from typing import List
from pathlib import Path

from trace.trace import Trace
from utils import open_data_trace_files, get_sentence_len, timer

@dataclass
class ArxivIOEvent():
    neighbors: List[int]

    @classmethod
    def from_trace(cls, items: List[str]):
        return cls([int(item) for item in items])


@dataclass
class ArxivTrace(Trace):
    events: List[ArxivIOEvent] = field(default_factory=list)

    def simulate(self, data_file_path: Path):
        """Run simulation on the storage for mol dataset.

        Args:
            data_file_path(Path): The data file path. Make sure the file is located in the storage
                                  that you want to simulate.

        """
        file_handlers = open_data_trace_files(data_file_path)
        sentence_len = get_sentence_len(file_handlers)

        self.handler_x, self.len_x = file_handlers["x"], sentence_len["x"]

        event_durations = []
        for event in self.events:
            event_durations.append(self._iterate_event(event))

        return event_durations

    @timer
    def _iterate_event(self, event):
        # Read node.x file
        self.handler_x.seek(0)
        self.handler_x.read(self.len_x * len(event.neighbors))

        # for neighbor in event.neighbors:
        #     self.handler_x.seek(neighbor * self.len_x)
        #     self.data_x = self.handler_x.read(self.len_x)


    @classmethod
    def parse_trace_file(cls, trace_file: str):
        """Parse the trace file.

        The trace file, for example, can be the access sequences of a GNN model.

        Args:
            trace_file(str): The trace file path.
            
        """
        events = []
        try:
            with open(trace_file, "r") as f:
                while True:
                    line = f.readline()
                    if not line:
                        break
                    # Assumption: The first line contains `#` in the beginning.
                    if "#" in line:
                        continue
                    events.append(ArxivIOEvent.from_trace(line.split(" ")))
        except:
            raise FileExistsError(f"Trace file({trace_file}) does not exist.")

        return cls(events)

    def __len__(self):
        return len(self.events)
