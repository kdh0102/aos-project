`simulate_ssd_trace.py` is to simulate ssd access time while running GNN inferences.

### How to run

```shell
python simulate_ssd_trace.py <TRACE_FILE> <DATASET_TYPE> <DATE_FILE_PATH>
```

1. TRACE_FILE: inference trace file. The file should include what nodes/edges are being accessed.
2. DATASET_TYPE: the dataset type. Currently, `arxiv` and `mol` are supported.
3. DATA_FILE_PATH: the data(e.g., node attribute) file path. The data should be located in the storage(e.g., SSD) you want to simulate.
