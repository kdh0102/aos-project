from  dataclasses import dataclass, field
from typing import List
from pathlib import Path

from trace.trace import Trace
from utils import sanity_check, open_data_trace_files, get_sentence_len, timer

@dataclass
class MolIOEvent():
    graph_idx: int
    node_start: int
    node_count: int
    edge_start: int
    edge_count: int
    y_value: int

    @classmethod
    def from_trace(cls, items: List[str]):
        return cls(*[int(item) for item in items])

@dataclass
class MolTrace(Trace):
    events: List[MolIOEvent] = field(default_factory=list)

    def simulate(self, data_file_path: Path):
        """Run simulation on the storage for mol dataset.

        Args:
            data_file_path(Path): The data file path. Make sure the file is located in the storage
                                  that you want to simulate.

        """
        sanity_check(data_file_path)
        file_handlers = open_data_trace_files(data_file_path)
        sentence_len = get_sentence_len(file_handlers)

        self.handler_x, self.len_x = file_handlers["x"], sentence_len["x"]
        self.handler_edge_attr, self.len_edge_attr = file_handlers["edge_attr"], sentence_len["edge_attr"]

        event_durations = []
        for event in self.events:
            event_durations.append(self._iterate_event(event))

        return event_durations

    @timer
    def _iterate_event(self, event):
        # Read node.x file
        # self.handler_x.seek(event.node_start * self.len_x)
        # self.data_x = self.handler_x.read(event.node_count * self.len_x)

        # self.handler_edge_attr.seek(event.edge_start * self.len_edge_attr)
        # self.data_edge_attr = self.handler_edge_attr.read(event.edge_count * self.len_edge_attr)

        self.data_x = self.handler_x[event.node_start * 36: (event.node_start + event.node_count) * 36]
        self.data_edge_attr = self.handler_edge_attr[event.edge_start * 12: (event.edge_start + event.edge_count) * 12]



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
                    events.append(MolIOEvent.from_trace(line.split("\t")))
        except:
            raise FileExistsError(f"Trace file({trace_file}) does not exist.")

        return cls(events)
