# 8-puzzle Game

1. Implementation
You will implement the following three algorithms as demonstrated in lecture. In particular:

* Breadth-First Search. Use an explicit queue.
* Depth-First Search. Use an explicit stack.
* A-Star Search. Use a priority queue. For the choice of heuristic, use the Manhattan 

priority function; that is, the sum of the distances of the tiles from their goal 
positions. Note that the blanks space is not considered an actual tile here.


2. Order of Visits
In this assignment, where an arbitrary choice must be made, we always visit child nodes 
in the "UDLR" order; that is, [‘Up’, ‘Down’, ‘Left’, ‘Right’] in that exact order. Specifically: 

* Breadth-First Search. Enqueue in UDLR order; dequeuing results in UDLR order.
* Depth-First Search. Push onto the stack in reverse-UDLR order; popping off results in UDLR order.
* A-Star Search. Since you are using a priority queue, what happens when there are 

duplicate keys? What do you need to do to ensure that nodes are retrieved from the 
priority queue in the desired order?