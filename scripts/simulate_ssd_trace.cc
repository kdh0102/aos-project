#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sys/mman.h>


void removeDupWord(std::string str, std::vector<std::string>& words) {
  std::string word = "";
  for (auto x : str) {
      if (x == ' ') {
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

struct ArxivEvent : Event {
  std::vector<int> neighbors;
};


class Trace {
 public:
  explicit Trace(std::string trace_file_path) {}
 
  void Simulate() {}

 protected:
  std::vector<Event> events;
};

class ArxivTrace : public Trace {
 public:
  explicit ArxivTrace(std::string trace_file_path) : Trace(trace_file_path) {
    std::ifstream trace_file(trace_file_path);

    if (not trace_file) {
      std::cout << "File does not exist.";
      exit(-1);
    }
    std::string s;
    while (trace_file) {
      getline(trace_file, s);
      if (s.find("#") == std::string::npos) {
        continue;
      }

      std::vector<std::string> words;
      removeDupWord(s, words);

      ArxivEvent event = ArxivEvent();
      for (auto& word : words) {
        std::cout << word << std::endl;
        event.neighbors.push_back(std::stoi(word));
      }
      events.push_back(event);
    }
    trace_file.close();
  }

  void Simulate() {
  }
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
    Trace trace = ArxivTrace(trace_file);
  }
  else if (dataset_type == "arxiv") {
    Trace trace = ArxivTrace(trace_file);
  }
  else {
    std::cout << dataset_type << " is not supported." << std::endl;
    exit(0);
  }

  return 0;
}
