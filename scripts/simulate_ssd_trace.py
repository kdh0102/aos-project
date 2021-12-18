import sys
from pathlib import Path

from trace.mol_trace import MolTrace


def dump_event_stats(event_times):
    print(f"# Events: {len(event_times)}")
    print(f"Avg Event Duration(ns): {sum(event_times) / len(event_times)}")
    print(f"Total Event Duration(ms): {sum(event_times) / 1e6}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise Exception("The arguments should include 1)trace file and 2)data file path.")

    trace_file = sys.argv[1]
    data_file_path = sys.argv[2]

    trace = MolTrace.parse_trace_file(trace_file)
    event_times = trace.simulate(Path(data_file_path))
    dump_event_stats(event_times)
