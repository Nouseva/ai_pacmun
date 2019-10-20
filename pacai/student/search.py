"""
In this file, you will implement generic search algorithms which are called by Pacman agents.
"""
from pacai.util.stack import Stack


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

    visit_set = set()
    visit_set.add(start)

    nextState = dfs_recurse(problem, visit_set, search_path, start)

    # Returns list with path to goal, empty if no valid path
    result = []
    while not (search_path.isEmpty()):
        result.append(search_path.pop())

    print("Path: %s" % str(result))
    return result


# Recursive calls through each element
def dfs_recurse(problem, visited, s_path, node):
    print("Checking node: %s" % (str(node)))
    visited.add(node)
    # Check if goal has been reached
    if problem.isGoal(node):
        return node

    successor_states = problem.successorStates(node)
    for s in successor_states:
        print("Checking node: %s" % (str(visited)))

        # check if successive state has been visited
        if s[0] in visited:
            continue

        print("%s not in set" % (str(s[0])))
        checked_path = dfs_recurse(problem, visited, s_path, s[0])

        # A path to goal has been found
        if checked_path:
            # Add found valid node, add direction of movement
            s_path.push(s[1])
            return node

    # There were no paths to goal from this node
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

