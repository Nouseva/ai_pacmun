"""
In this file, you will implement generic search algorithms which are called by Pacman agents.
"""
from pacai.util.stack import Stack
from pacai.util.queue import Queue
from pacai.util.priorityQueue import PriorityQueue


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first [p 85].

    Your search algorithm needs to return a list of actions that reaches the goal.
    Make sure to implement a graph search algorithm [Fig. 3.7].

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    ```
    print("Start: %s" % (str(problem.startingState())))
    print("Is the start a goal?: %s" % (problem.isGoal(problem.startingState())))
    print("Start's successors: %s" % (problem.successorStates(problem.startingState())))
    ```
    """

    # *** Your Code Here ***

    start = problem.startingState()
    visited = set()
    directions = Stack()
    r = []

    dfs_recurse(problem, visited, directions, start)

    while not (directions.isEmpty()):
        r.append(directions.pop())
    print("Path: %s" % str(r))

    return r

def visit_valid(problem, visited, storage, node):
    visited.add(node)
    visit_count = 0

    successor_states = problem.successorStates(node)
    for s in successor_states:
        if s[0] in visited:
            continue
        storage.push(s)
        # Add one for each valid state resulting from this node
        visit_count += 1

    return visit_count


# Recursive calls through each element
def dfs_recurse(problem, visited, directions, node):

    fringe = Stack()
    num_visited = visit_valid(problem, visited, fringe, node)

    # If there is no visited subnodes, node is dead-end
    if num_visited == 0:
        return None

    while not (fringe.isEmpty()):
        current = fringe.pop()
        if (problem.isGoal(current[0])):
            directions.push(current[1])
            return node

        next_state = dfs_recurse(problem, visited, directions, current[0])
        if next_state:
            # Current state is on the path to goal
            # Store direction of movement
            directions.push(current[1])
            return node

    return None


def visit_valid_bfs(problem, visited, parent_map, storage, node):
    visited.add(node)
    visit_count = 0

    successor_states = problem.successorStates(node)
    for s in successor_states:
        if s[0] in visited:
            continue
        visited.add(s[0])
        storage.push(s)
        visit_count += 1

        # keep track of parent of s[0]
        parent_map[s[0]] = (node, s[1])
    return visit_count

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first. [p 81]
    """

    start = problem.startingState()
    visited_set = set()
    parent_map = {start: None}
    directions = Stack()
    fringe = Queue()
    goal = None
    result = []

    visit_valid_bfs(problem, visited_set, parent_map, fringe, start)

    # Form the parent tree for all nodes
    while not(fringe.isEmpty()):
        current = fringe.pop()

        if problem.isGoal(current[0]):
            goal = current[0]
            break
        visit_valid_bfs(problem, visited_set, parent_map, fringe, current[0])

    parent = parent_map[goal]
    while parent:
        directions.push(parent[1])
        parent = parent_map[parent[0]]

    while not (directions.isEmpty()):
        result.append(directions.pop())

#    print("Path: %s" % (str (result)))
    return result

def visit_valid_ucs(problem, visited, parent_map, storage, node, dist):
    visited.add(node)
    visit_count = 0

    successor_states = problem.successorStates(node)
    for s in successor_states:
        if s[0] in visited:
            continue

        # Push onto fringe, child and total distance up until it
        visited.add(s[0])
        # distance is stored as part of the item
        storage.push((s, dist + s[2]), dist + s[2])
        visit_count += 1

        # keep track of parent of s[0]
        parent_map[s[0]] = (node, s[1])
    return visit_count

def uniformCostSearch(problem):
    """
    Search the node of least total cost first.
    """

    start = problem.startingState()
    visited_set = set()
    parent_map = {start: None}
    directions = Stack()
    fringe = PriorityQueue()
    result = []
    goal = None
    distance = 0

    visit_valid_ucs(problem, visited_set, parent_map, fringe, start, distance)

    while not (fringe.isEmpty()):
        current, distance = fringe.pop()

        if problem.isGoal(current[0]):
            goal = current[0]
            break
        visit_valid_ucs(problem, visited_set, parent_map, fringe, current[0], distance)

    parent = parent_map[goal]
    while parent:
        directions.push(parent[1])
        parent = parent_map[parent[0]]

    while not (directions.isEmpty()):
        result.append(directions.pop())

    return result

def visit_valid_aStar(problem, visited, parent_map, storage, node, dist, heu):
    visited.add(node)
    visit_count = 0

    successor_states = problem.successorStates(node)
    for s in successor_states:
        if s[0] in visited:
            continue

        visited.add(s[0])
        # Push onto fringe, child and distance to goal
        # find the estimated distance to goal
        h = heu(s[0], problem)
        # actual distance is stored as part of the item
        storage.push((s, dist + s[2]), dist + h)
        visit_count += 1

        # keep track of parent of s[0]
        parent_map[s[0]] = (node, s[1])
    return visit_count


def aStarSearch(problem, heuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """
    start = problem.startingState()
    visited_set = set()
    parent_map = {start: None}
    directions = Stack()
    fringe = PriorityQueue()
    result = []
    goal = None
    distance = 0

    visit_valid_aStar(problem, visited_set, parent_map, fringe, start, distance, heuristic)

    while not (fringe.isEmpty()):
        current, distance = fringe.pop()

        if problem.isGoal(current[0]):
            goal = current[0]
            break
        visit_valid_aStar(problem, visited_set, parent_map, fringe, current[0], distance, heuristic)

    parent = parent_map[goal]
    while parent:
        directions.push(parent[1])
        parent = parent_map[parent[0]]

    while not (directions.isEmpty()):
        result.append(directions.pop())

    return result

    # *** Your Code Here ***
    raise NotImplementedError()
