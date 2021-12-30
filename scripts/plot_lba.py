import sys
from dataclasses import dataclass
from matplotlib import pyplot as plt
from pathlib import Path


@dataclass
class BlkEvent:
    seq_id: int
    time_stamp: float
    event_type: str
    RWBS: str
    start_blk: int
    num_blk: int
    is_python: bool

    @classmethod
    def from_items(cls, items):
        return cls(
            int(items[2]),
            float(items[3]),
            items[5],
            items[6],
            int(items[7]),
            int(items[9]),
            True if "python" in items[10] else False
        )


def is_end_of_trace(line):
    if "CPU" in line:
        return True
    else:
        return False


def parse_blktrace(file_name):
    blkevents = []
    with open(file_name, "r") as f:
        while line := f.readline():
            if is_end_of_trace(line):
                break

            items = [item for item in line.split(" ") if item]

            # check blktrace event
            if items[5] != "D" or items[6] != "R" or "python" not in items[10]:
                continue

            blkevents.append(BlkEvent.from_items(items))
    return blkevents


def get_timestamps(blkevents):
    timestamps = []
    for blkevent in blkevents:
        timestamps.append(blkevent.time_stamp * 1000)

    return timestamps


def get_blk_addr(blkevents):
    blk_addrs = []
    for blkevent in blkevents:
        blk_addrs.append(blkevent.start_blk)

    return blk_addrs


def plot_lba_heatmap(blkevents, file_name):
    fig, ax = plt.subplots()
    ax.grid(axis='y', linestyle='--')
    ax.set_axisbelow(True)
    ax.plot(get_timestamps(blkevents), get_blk_addr(blkevents), marker='o', color="#5cb85c", linestyle="", markersize=2)

    ax.set_ylabel('Memory Address (Byte)', fontsize=20, fontweight='bold')
    ax.set_xlabel('Access Time(ms)', fontsize=20, fontweight='bold')

    plt.savefig(f"{file_name.parent}/lba_heatmap_{file_name.name}.png", bbox_inches='tight')
    plt.clf()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Please pass trace path.")
        exit(0)

    trace_path = Path(sys.argv[1])

    for file_name in trace_path.iterdir():
        if file_name.is_file() and ".eps" not in file_name.name:
            blkevents = parse_blktrace(file_name)
            plot_lba_heatmap(blkevents, file_name)
