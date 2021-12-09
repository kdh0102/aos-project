import sys
from dataclasses import dataclass, field
from typing import List


@dataclass
class IOEvent:
    """A class to represent an IO event."""

    @classmethod
    def from_trace(cls, items: List[str]):
        return cls(*[int(item) for item in items])

@dataclass
class MolIOEvent(IOEvent):
    graph_idx: int
    node_start: int
    node_count: int
    edge_start: int
    edge_count: int
    y_value: int


@dataclass
class Trace:
    events: List[IOEvent] = field(default_factory=list)

    def simulate(self, file_name: str):
        """Run simulation on the stoarge.

        Args:
            file_name(str): The data file name. Make sure the file is located in the storage
                            that you want to simulate.

        """
        try:
            with open(file_name, "r") as f:
                for event in self.events:
                    f.seek(event.offset)
                    if event.type == "r":
                        f.read(event.size)
                    else:
                        raise NotImplemented

        except:
            raise FileExistsError(f"{file_name} does not exist.")


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


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise Exception("The arguments should include 1)file path and 2)trace file.")

    trace_file = sys.argv[1]
    data_file_path = sys.argv[2]

    trace = Trace.parse_trace_file(trace_file)
    trace.simulate(data_file_path)
