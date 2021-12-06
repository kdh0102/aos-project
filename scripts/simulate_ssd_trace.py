import sys
from dataclasses import dataclass, field
from typing import List


@dataclass
class IOEvent:
    """
    A class to represent an IO event.

    offset(int): Where to start the event from the file (bytes).
    size(int): The number of bytes of the event.
    type(str): The type of the event (e.g., read, write).

    """
    offset: int
    size: int
    type: str


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
        traces = cls()
        try:
            with open(trace_file, "r") as f:
                pass
        except:
            raise FileExistsError(f"Trace file({file_name}) does not exist.")

        return traces


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise Exception("The arguments should include 1)file path and 2)trace file.")

    file_name = sys.argv[1]
    trace_file = sys.argv[2]

    trace = Trace.parse_trace_file(trace_file)
    trace.simulate(file_name)
