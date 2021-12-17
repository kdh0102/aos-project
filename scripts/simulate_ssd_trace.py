import sys
from multiprocessing import Process
from subprocess import call
from dataclasses import dataclass, field
from typing import List
from pathlib import Path


SPLIT_FILES = {
    "test": "test_idx.txt",
    "train": "train_idx.txt",
    "test": "test_idx.txt"
}

DATA_TRACE_FILES = {
    "x": "data.x.txt",
    "y": "data.y.txt",
    "edge_attr": "data.edge_attr.txt",
    "edge_index": "data.edge_index.txt",
    "edge_index_row": "data.edge_index_row.txt",
    "edge_index_col": "data.edge_index_col.txt"
}

STORAGE = "/dev/nvme0n1"

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
class Blktrace:
    BLKTRACE = "blktrace"

    @staticmethod
    def run() -> Process:
        def call_blktrace():
            call(["sudo", Blktrace.BLKTRACE, "-d", STORAGE])

        p = Process(target=call_blktrace, daemon=True)
        return p


def record_blktrace(func):
    def blktrace(*args, **kwargs):
        p = Blktrace.run()
        p.start()
        result = func(*args, **kwargs)
        p.kill()
        p.join()
        return result
    return blktrace


def sanity_check(data_file_path: Path):
    for file in {**SPLIT_FILES, **DATA_TRACE_FILES}.values():
        assert((data_file_path / file).exists())


def open_data_trace_files(data_file_path):
    handlers = {}
    for key, fname in DATA_TRACE_FILES.items():
        handlers[key] = open(data_file_path / fname, "r")

    return handlers


def get_sentence_len(handlers):
    sentence_len = {}
    for key, file_handler in handlers.items():
        sentence_len[key] = len(file_handler.readline())
        file_handler.seek(0)

    return sentence_len


@dataclass
class Trace:
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

        handler_x, len_x = file_handlers["x"], sentence_len["x"]
        handler_edge_attr, len_edge_attr = file_handlers["edge_attr"], sentence_len["edge_attr"]
        handler_edge_index_row, len_edge_index_row = file_handlers["edge_index_row"], sentence_len["edge_index_row"]
        handler_edge_index_col, len_edge_index_col = file_handlers["edge_index_col"], sentence_len["edge_index_col"]

        for event in self.events:
            # Read node.x file
            handler_x.seek(event.node_start * len_x)
            data_x = handler_x.read(event.node_count * len_x)

            handler_edge_attr.seek(event.edge_start * len_edge_attr)
            data_edge_attr = handler_edge_attr.read(event.edge_count * len_edge_attr)

            handler_edge_index_row.seek(event.edge_start * len_edge_index_row)
            data_edge_index_row = handler_edge_index_row.read(event.edge_count * len_edge_index_row)

            handler_edge_index_col.seek(event.edge_start * len_edge_index_col)
            data_edge_index_col = handler_edge_index_col.read(event.edge_count * len_edge_index_col)


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
        raise Exception("The arguments should include 1)trace file and 2)data file path.")

    trace_file = sys.argv[1]
    data_file_path = sys.argv[2]

    trace = Trace.parse_trace_file(trace_file)
    trace.simulate(Path(data_file_path))
