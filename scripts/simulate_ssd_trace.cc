#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>
#include <sys/mman.h>

#define STORAGE "/dev/nvme0n1"
#define DATA_X "data.x"
#define EDGE_ATTR "data.edge_attr"

void removeDupWord(std::string str, std::vector<std::string>& words) {
  std::string word = "";
  for (auto x : str) {
      if (x == ' ' || x == '\t') {
        if (word.size() > 0) {
          words.push_back(word);
        }
        word = "";
      } else {
        word = word + x;
      }
  }
  if (word.size() > 0) {
    words.push_back(word);
  }
}

struct Event {};

struct MolEvent : Event {
  int graph_idx;
  int node_start;
  int node_count;
  int edge_start;
  int edge_count;
  int y_value;

  MolEvent(std::vector<std::string> words) {
    if (words.size() != 6) {
      std::cout << "Check Mol event format." << std::endl;
      exit(-1);
    }
    graph_idx = std::stoi(words[0]);
    node_start = std::stoi(words[1]);
    node_count = std::stoi(words[2]);
    edge_start = std::stoi(words[3]);
    edge_count = std::stoi(words[4]);
    y_value = std::stoi(words[5]);
  }
};

struct ArxivEvent : Event {
  std::vector<int> neighbors;

  ArxivEvent(std::vector<std::string> words) {
    for (auto& word : words) {
      neighbors.push_back(std::stoi(word));
    }
  }
};

class Trace {
 public:
  explicit Trace(std::string data_file_path) {
    OpenDataFiles(data_file_path);
  }
  std::vector<Event>& GetEvents() { return events; }
  void OpenDataFiles(std::string data_file_path) {
    std::vector<std::string> files = {DATA_X, EDGE_ATTR};

    for (auto file : files) {
      std::string file_name = data_file_path + file + ".bin";
      std::cout << "file name : " + file_name << std::endl;
      std::ifstream file_handler(file_name, std::ifstream::binary);
      if (file_handler) {
        data_files[file] = std::move(file_handler);
      } else {
        std::cout << file + " does not exist." << std::endl;
      }
    }
  }
  virtual void Simulate() = 0;
  virtual Event CreateEvent(std::vector<std::string> words) = 0;

 protected:
  std::vector<Event> events;
  std::map<std::string, std::ifstream> data_files;
};

class MolTrace : public Trace {
 public:
  explicit MolTrace(std::string data_file_path) : Trace(data_file_path) {}
  Event CreateEvent(std::vector<std::string> words) override {
    return MolEvent(words);
  }

  void Simulate() {
  }
};

class ArxivTrace : public Trace {
 public:
  explicit ArxivTrace(std::string data_file_path) : Trace(data_file_path) {}
  Event CreateEvent(std::vector<std::string> words) override {
    return ArxivEvent(words);
  }

  void Simulate() {
  }
};

void FromTraceFile(Trace& trace, std::string trace_file_path) {
  std::ifstream trace_file(trace_file_path);
  if (not trace_file) {
    std::cout << "file does not exist.";
    exit(-1);
  }

  std::string s;
  while (trace_file) {
    getline(trace_file, s);
    if (s.find("#") != std::string::npos || s.size() == 0) {
      continue;
    }
    std::vector<std::string> words;
    removeDupWord(s, words);

    trace.GetEvents().push_back(trace.CreateEvent(words));
  }
  trace_file.close();
};

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cout << "Three arguments should be passed." << std::endl;
    exit(0);
  }

  std::string trace_file = argv[1];
  std::string dataset_type = argv[2];
  std::string data_file_path = argv[3];

  if (dataset_type == "mol") {
    MolTrace trace = MolTrace(data_file_path);
    FromTraceFile(trace, trace_file);
    trace.Simulate();
  } else if (dataset_type == "arxiv") {
    ArxivTrace trace = ArxivTrace(data_file_path);
    FromTraceFile(trace, trace_file);
    trace.Simulate();
  } else {
    std::cout << dataset_type << " is not supported." << std::endl;
    exit(0);
  }

  return 0;
}
