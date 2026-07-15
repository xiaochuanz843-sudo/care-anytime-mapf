#include "node.h"

int Node::cntIndex = 0;

Node::Node(int _id) : id(_id), index(cntIndex) {
  ++cntIndex;
  pos = Vec2f(0, 0);
}
