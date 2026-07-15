

#pragma once
#include "grid.h"

class SimpleGrid : public Grid {
protected:
  std::string filename;

  void init();
  void setBasicParams();
  void createNodes();
  void createEdges();
  virtual void setStartGoal();

public:
  SimpleGrid(std::string _filename);
  SimpleGrid(std::string _filename, std::mt19937* _MT);
  ~SimpleGrid();

  // for iterative MAPF
  virtual Node* getNewGoal(Node* v);

  std::string getMapName() { return filename; }

  std::string logStr();
};
