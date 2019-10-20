"""
In this file, you will implement generic search algorithms which are called by Pacman agents.
"""
from pacai.util.stack import Stack
from pacai.core.actions import Actions


def travel_direction(final_pos, init_pos):
    fx, fy = final_pos
    ix, iy = init_pos
    return (fx - ix, fy - iy)


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
    print("Start: %s" % (str(problem.startingState())))
    print("Is the start a goal?: %s" % (problem.isGoal(problem.startingState())))
    print("Start's successors: %s" % (problem.successorStates(problem.startingState())))

    search_path = Stack()
    start = problem.startingState()
    visited = set()
    result = Stack()
    r = []

    next_state = dfs_recurse(problem, visited, result, start)
    direct = Actions.vectorToDirection(travel_direction(next_state, start))
    result.push(direct)

    while not (result.isEmpty()):
        r.append(result.pop())
    print("Path: %s" % str(r))

    return r



def visit_valid(problem, visited, s_stack, node):
    print("Visiting Node: %s" % (str(node)))
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
    print("Checking node: %s" % (str(node)))

    s_path = Stack()
    visits = visit_valid(problem, visited, s_path, node)

    # If there is no visited subnodes
    if visits == 0:
        # Current node is goal
        if problem.isGoal(node):
            return node
        # Or current node is dead-end
        else:
            return None

    while not (s_path.isEmpty()):
        to_check = s_path.pop()

        next_state = dfs_recurse(problem, visited, directions, to_check[0])
        if next_state:
            # Current state is on the path to goal
            # Store direction of movement
            directions.push(Actions.vectorToDirection(travel_direction(next_state, to_check[0])))
            return to_check[0]

    return None


def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first. [p 81]
    """

    # *** Your Code Here ***
    raise NotImplementedError()

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

