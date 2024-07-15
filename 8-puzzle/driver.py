import sys
import argparse
import timeit
import resources
from collections import deque
from heapq import heappush, heappop, heapify
import itertools

class State(object):
    def __init__(self, state, parent, move, depth, cost, hx):
        self.state = state
        self.parent = parent
        self.move = move
        self.depth = depth
        self.cost = cost
        self.hx = hx
        if self.state:
            self.map = ''.join(str(e) for e in self.state)
    def __eq__(self, other):
        return self.map == other.map
    def __lt__(self, other):
        return self.map < other.map

goal_puzzle = [0,1,2,3,4,5,6,7,8]
goal = State
initvalue = list()
puzzle_len = 0
end_column = 0
nodes_expanded = 0
last_node_depth = 0
max_search_depth = 0
root_node = 0
pathmovement = list()
costs = set()

def run_function():
    global function, stqu
    parser = argparse.ArgumentParser()
    parser.add_argument('ch', type = str, choices=['bfs', 'dfs', 'ast'])
    parser.add_argument('node')
    args = parser.parse_args()
    read_node(args.node)
    choose = args.ch
    if args.ch == 'bfs':
        function = bfs
    elif args.ch == 'dfs':
        function = dfs
    else:
        function = ast
    start = timeit.default_timer()
    frontier = function(initvalue)
    stop = timeit.default_timer()
    fwrite(frontier, stop-start)

def checkvisited(stqu, node, visited):
    global max_search_depth
    if (function == bfs or function == dfs):
        for treenode in node:
            if treenode.map not in visited:
                stqu.append(treenode)
                visited.add(treenode.map)
                if treenode.depth > max_search_depth:
                    max_search_depth += 1
        return node
    elif (function == ast):
        for treenode in node:
            treenode.hx = treenode.cost + ab_value(treenode.state)
            stqu = (treenode.hx, treenode.move, treenode)
            if treenode.map not in visited:
                heappush(heap, stqu)
                visited.add(treenode.map)
                heap_entry[treenode.map] = stqu
                if treenode.depth > max_search_depth:
                    max_search_depth += 1
            elif treenode.map in heap_entry and treenode.hx < heap_entry[treenode.map][2].hx:
                hindex = heap.index((heap_entry[treenode.map][2].hx, heap_entry[treenode.map][2].move, heap_entry[treenode.map][2]))
                heap[int(hindex)] = stqu
                heap_entry[treenode.map] = stqu
                heapify(heap)
        return node

def ab_value(state):
    for b, g in ((state.index(i), goal_puzzle.index(i)) for i in range(1, puzzle_len)):
        MX = abs(b % end_column - g % end_column)
        MY = abs(int(b/end_column - g/end_column))
        dist = MX + MY
        return dist

def expandnode(node):
    global nodes_expanded
    changestate = node.state
    current_status = changestate[0:9]
    zeroindex = current_status.index(0)
    nodes_expanded += 1
    children_node = list()
    if(zeroindex-3 >= 0):
        children_node.append(State(swap(node.state, zeroindex, zeroindex-3), node, 'Up', node.depth + 1, node.cost + 1, 0))
    if(zeroindex+3 <= 8):
        children_node.append(State(swap(node.state, zeroindex, zeroindex+3), node, 'Down', node.depth + 1, node.cost + 1, 0))
    if(zeroindex%3-1 >= 0):
        children_node.append(State(swap(node.state, zeroindex, zeroindex-1), node, 'Left', node.depth + 1, node.cost + 1, 0))
    if(zeroindex%3+1 < 3):
        children_node.append(State(swap(node.state, zeroindex, zeroindex+1), node, 'Right', node.depth + 1, node.cost + 1, 0))
    nodes = [child for child in children_node if child.state]
    return nodes

def swap(state, z, newnode):
    temp = state[:]
    temp[z] = state[newnode]
    temp[newnode] = 0
    return temp

def path():
    previous_node = goal
    while initvalue != previous_node.state:
        #pathmovement.insert(0, previous_node.state) #number movement check
        pathmovement.insert(0, previous_node.move) #move path check
        previous_node = previous_node.parent
    return pathmovement

def bfs(start_state):
    global last_node_depth, goal
    visited = set()
    Q = deque([State(start_state, None, None, 0, 0, 0)])
    while Q:
        node = Q.popleft()
        visited.add(node.map)
        if node.state == goal_puzzle:
            goal = node
            return Q
        else:
            child_nodes = expandnode(node)
            checkvisited(Q, child_nodes, visited)
            if len(Q) > last_node_depth:
                last_node_depth = len(Q)

def dfs(start_state):
    global last_node_depth, goal
    visited = set()
    stack = list([State(start_state, None, None, 0, 0, 0)])
    while stack:
        node = stack.pop()
        if node.state == goal_puzzle:
            goal = node
            return stack
        else:
            visited.add(node.map)
            child_nodes = reversed(expandnode(node))
            checkvisited(stack, child_nodes, visited)
            if len(stack) > last_node_depth:
                last_node_depth = len(stack)

def ast(start_state):
    global last_node_depth, goal, max_search_depth, heap, heap_entry
    visited = set()
    heap = list()
    heap_entry = {}
    counter = itertools.count()
    hx = ab_value(start_state)
    root_node = State(start_state, None, None, 0, 0, hx)
    entry = (hx, 0, root_node)
    heappush(heap, entry)
    heap_entry[root_node.map] = entry
    while heap:
        node = heappop(heap)
        visited.add(node[2].map)
        if node[2].state == goal_puzzle:
            goal = node[2]
            return heap
        else:
            child_nodes = expandnode(node[2])
            checkvisited(entry, child_nodes, visited)
            if len(heap) > last_node_depth:
                last_node_depth = len(heap)

def read_node(configuration):
    global puzzle_len, end_column
    data = configuration.split(",")
    for element in data:
        initvalue.append(int(element))
    puzzle_len = len(initvalue)
    end_column = int(puzzle_len ** 0.5)

def fwrite(frontier, time):
    global pathmovement
    max_ram_usage = 0
    if sys.platform == "win32":
        import psutil
        max_ram_usage = psutil.Process().memory_info().rss/1024/1024
    else:
        import resource
        max_ram_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024/1024
    pathmovement = path()
    
    # if you wnat see direct result than you use this part
    """
    print("path_to_goal: " + str(pathmovement))
    print("cost_of_path: " + str(len(pathmovement)))
    print("nodes_expanded: " + str(nodes_expanded))
    print("search_depth: " + str(goal.depth))
    print("max_search_depth: " + str(max_search_depth))
    print("running_time: " + format(time, '.8f'))
    print('mmax_ram_usage: ' + str(max_ram_usage))
    """
    f = open('output.txt', 'w')
    f.write("path_to_goal: " + str(pathmovement))
    f.write("\ncost_of_path: " + str(len(pathmovement)))
    f.write("\nnodes_expanded: " + str(nodes_expanded))
    f.write("\nsearch_depth: " + str(goal.depth))
    f.write("\nmax_search_depth: " + str(max_search_depth))
    f.write("\nrunning_time: " + format(time, '.8f'))
    f.write('\nmmax_ram_usage: ' + str(max_ram_usage))
    f.close()

if __name__ == '__main__':
    run_function()
