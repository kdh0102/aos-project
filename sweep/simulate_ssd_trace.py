import sys
from pathlib import Path

from trace.trace import Trace
from trace.mol_trace import MolTrace
from trace.arxiv_trace import ArxivTrace


def dump_event_stats(event_times):
    print(f"# Events: {len(event_times)}")
    print(f"Avg Event Duration(ns): {sum(event_times) / len(event_times)}")
    print(f"Total Event Duration(ms): {sum(event_times) / 1e6}")


if __name__ == "__main__":
    # if len(sys.argv) != 4:
    #     raise Exception("The arguments should include 1)trace file 2)dataset type and 3)data file path.")

    trace_name = sys.argv[1]
    # dataset_type = sys.argv[2]
    # data_file_path = sys.argv[3]

    # trace = Trace()
    # if dataset_type == "mol":
    #     trace = MolTrace.parse_trace_file(trace_file)
    # elif dataset_type == "arxiv":
    #     trace = ArxivTrace.parse_trace_file(trace_file)
    # else:
    #     raise NotImplementedError

    trace_file = "../examples/nodeproppred/arxiv/" + trace_name
    trace = ArxivTrace.parse_trace_file(trace_file)

    data_file_path = "../examples/nodeproppred/arxiv"

    event_times = trace.simulate(Path(data_file_path))
    dump_event_stats(event_times)
