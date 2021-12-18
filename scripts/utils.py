import time
from pathlib import Path
from dataclasses import dataclass
from multiprocessing import Process
from subprocess import call

from constants import SPLIT_FILES, DATA_TRACE_FILES, STORAGE


def timer(func):
    # a decorator to record the duration
    # Note that this decorator will return the function duration, instead of the function output.
    def _timer(*args, **kwargs):
        start = time. time_ns()
        func(*args, **kwargs)
        end = time. time_ns()
        return end - start
    return _timer


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
        if (data_file_path / fname).exists():
            handlers[key] = open(data_file_path / fname, "r")

    return handlers


def get_sentence_len(handlers):
    sentence_len = {}
    for key, file_handler in handlers.items():
        sentence_len[key] = len(file_handler.readline())
        file_handler.seek(0)

    return sentence_len
