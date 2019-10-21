"""
In this file, you will implement generic search algorithms which are called by Pacman agents.
"""
from pacai.util.stack import Stack
from pacai.util.queue import Queue


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
 #   print("Start: %s" % (str(problem.startingState())))
 #   print("Is the start a goal?: %s" % (problem.isGoal(problem.startingState())))
 #   print("Start's successors: %s" % (problem.successorStates(problem.startingState())))

    start = problem.startingState()
    visited = set()
    result = Stack()
    r = []

    next_state = dfs_recurse(problem, visited, result, start)

    while not (result.isEmpty()):
        r.append(result.pop())
    print("Path: %s" % str(r))

    return r



def visit_valid(problem, visited, s_stack, node):
#    print("Visiting Node: %s" % (str(node)))
    visited.add(node)
    visit_count = 0

    successor_states = problem.successorStates(node)
    for s in successor_states:
        if s[0] in visited:
            continue
        s_stack.push(s)
        # Add one for each valid state resulting from this node
        visit_count += 1

    return visit_count
#    dfs_recurse(problem, visited, result, s_path, node)


# Recursive calls through each element
def dfs_recurse(problem, visited, directions, node):
#    print("Checking node: %s" % (str(node)))

    s_path = Stack()
    visits = visit_valid(problem, visited, s_path, node)

    # If there is no visited subnodes, node is dead-end
    if visits == 0:
        return None

    while not (s_path.isEmpty()):
        to_check = s_path.pop()
        if (problem.isGoal(to_check[0])):
            directions.push(to_check[1])
            return node

        next_state = dfs_recurse(problem, visited, directions, to_check[0])
        if next_state:
            # Current state is on the path to goal
            # Store direction of movement
            directions.push(to_check[1])
            return node

    return None


def visit_valid_bfs(problem, visited, parent_map, storage, node):
    visited.add(node)
    visit_count = 0

    successor_states = problem.successorStates(node)
    for s in successor_states:
        if s[0] in visited:
            continue
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
    to_check = Queue()
    goal = None
    result = []

    visit_valid_bfs(problem, visited_set, parent_map, to_check, start)

    # Form the parent tree for all nodes
    while not(to_check.isEmpty()):
        current = to_check.pop()

        if problem.isGoal(current[0]):
            goal = current[0]
        visit_valid_bfs(problem, visited_set, parent_map, to_check, current[0])

    parent = parent_map[goal]
    while parent:
        directions.push(parent[1])
        parent = parent_map[parent[0]]

    while not (directions.isEmpty()):
        result.append(directions.pop())

#    print("Path: %s" % (str (result)))
    return result


def uniformCostSearch(problem):
    """
    Search the node of least total cost first.
    """

    # *** Your Code Here ***
    raise NotImplementedError()

def aStarSearch(problem, heuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """

    # *** Your Code Here ***
    raise NotImplementedError()

