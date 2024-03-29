"""
This file contains incomplete versions of some agents that can be selected to control Pacman.
You will complete their implementations.

Good luck and happy searching!
"""

import logging

from pacai.core.actions import Actions
from pacai.core.directions import Directions
from pacai.core import distance
from pacai.core.search.position import PositionSearchProblem
from pacai.core.search.problem import SearchProblem
from pacai.agents.base import BaseAgent
from pacai.agents.search.base import SearchAgent
from pacai.student.search import uniformCostSearch
from pacai.util import util


class CornerState:
    def __init__(self, destinationPosition, toVisit):
        self._position = destinationPosition
        self._visitTargets = list(toVisit)
        if destinationPosition in toVisit:
            self._visitTargets.remove(destinationPosition)

        self._numTargets = len(self._visitTargets)

        self._hash = None

    def __hash__(self):
        if (self._hash is None):
            self._hash = util.buildHash(self._position, self._numTargets, *self._visitTargets)

        return self._hash

    def __list__(self):
        return [self._position, self._visitTargets]

    def __str__(self):
        result = "{" + str(self._position) + "," + str(self._visitTargets) + "}"
        return result

    def __eq__(self, other):
        if self._position != other._position:
            return False
        if self._numTargets != other._numTargets:
            return False
        if self._visitTargets != other._visitTargets:
            return False
        return True

    def __lt__(self, other):
        return self._numTargets < other._numTargets

    def getAgentPosition(self):
        return self._position

    def getTargets(self):
        return self._visitTargets

    def getTargetCount(self):
        return self._numTargets

class CornersProblem(SearchProblem):
    """
    This search problem finds paths through all four corners of a layout.

    You must select a suitable state space and successor function.
    See the `pacai.core.search.position.PositionSearchProblem` class for an example of
    a working SearchProblem.

    Additional methods to implement:

    `pacai.core.search.problem.SearchProblem.startingState`:
    Returns the start state (in your search space,
    NOT a `pacai.core.gamestate.AbstractGameState`).

    """
    def __init__(self, startingGameState):
        super().__init__()

        self.walls = startingGameState.getWalls()
        top = self.walls.getHeight() - 2
        right = self.walls.getWidth() - 2

        self.corners = ((1, 1), (1, top), (right, 1), (right, top))
        for corner in self.corners:
            if not startingGameState.hasFood(*corner):
                logging.warning('Warning: no food in corner ' + str(corner))

        # *** Your Code Here ***
        self.startingPosition = startingGameState.getPacmanPosition()
        self._visitedLocations = set()
        self._visitHistory = []
        self._numExpanded = 0
        self.goal = [CornerState((1, 1), ()), CornerState((1, top), ()),
                CornerState((right, 1), ()), CornerState((right, top), ())]

    def getExpandedCount(self):
        return self._numExpanded

    def getVisitHistory(self):
        return self._visitHistory

    def updateVisitHistory(self, pos):
        self._visitedLocations.add(pos)
        self._visitHistory.append(pos)

    def actionsCost(self, actions):
        """
        Returns the cost of a particular sequence of actions.
        If those actions include an illegal move, return 999999.
        This is implemented for you.
        """
        if (actions is None):
            return 999999

        x, y = self.startingPosition
        for action in actions:
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return 999999

        return len(actions)

    def startingState(self):
        return CornerState(self.startingPosition, self.corners)

    def successorStates(self, state):
        currentPosition = state.getAgentPosition()
        self._numExpanded += 1
        self.updateVisitHistory(currentPosition)

        successors = []

        for action in Directions.CARDINAL:
            x, y = currentPosition
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            hitsWall = self.walls[nextx][nexty]

            if (not hitsWall):
                dest = (nextx, nexty)
                cState = CornerState(dest, state.getTargets())
                successors.append((cState, action, 1))

        return successors

    """
    Returns whether this search state is a goal state of the problem.
    Goal is reached when there is no more corners to reach
    """
    def isGoal(self, state):
        return state.getTargetCount() == 0

def cornersHeuristic(state, problem):
    """
    A heuristic for the CornersProblem that you defined.

    This function should always return a number that is a lower bound
    on the shortest path from the state to a goal of the problem;
    i.e. it should be admissible.
    (You need not worry about consistency for this heuristic to receive full credit.)
    """

    # Useful information.
    # corners = problem.corners  # These are the corner coordinates
    # walls = problem.walls  # These are the walls of the maze, as a Grid.

    # *** Your Code Here ***

    targets = state.getTargets()
    pos = state.getAgentPosition()
    dist = []

    for t in targets:
        dist.append(distance.manhattan(t, pos))
        # dist.append(distance.euclidean(t, pos))

    if dist:
        return min(dist) + state.getTargetCount()
    return 0

def foodHeuristic(state, problem):
    """
    Your heuristic for the FoodSearchProblem goes here.

    This heuristic must be consistent to ensure correctness.
    First, try to come up with an admissible heuristic;
    almost all admissible heuristics will be consistent as well.

    If using A* ever finds a solution that is worse than what uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!
    On the other hand, inadmissible or inconsistent heuristics may find optimal solutions,
    so be careful.

    The state is a tuple (pacmanPosition, foodGrid) where foodGrid is a
    `pacai.core.grid.Grid` of either True or False.
    You can call `foodGrid.asList()` to get a list of food coordinates instead.

    If you want access to info like walls, capsules, etc., you can query the problem.
    For example, `problem.walls` gives you a Grid of where the walls are.

    If you want to *store* information to be reused in other calls to the heuristic,
    there is a dictionary called problem.heuristicInfo that you can use.
    For example, if you only want to count the walls once and store that value, try:
    ```
    problem.heuristicInfo['wallCount'] = problem.walls.count()
    ```
    Subsequent calls to this heuristic can access problem.heuristicInfo['wallCount'].
    """

    position, foodGrid = state

    # *** Your Code Here ***
    pellet_list = foodGrid.asList()
    dist = set()

    for p in pellet_list:
        dist.add(distance.manhattan(p, position))

    if dist:
        return min(dist) + foodGrid.count()
    return 0


class ClosestDotSearchAgent(SearchAgent):
    """
    Search for all food using a sequence of searches.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index)

    def registerInitialState(self, state):
        self._actions = []
        self._actionIndex = 0

        currentState = state

        while (currentState.getFood().count() > 0):
            nextPathSegment = self.findPathToClosestDot(currentState)  # The missing piece
            self._actions += nextPathSegment

            for action in nextPathSegment:
                legal = currentState.getLegalActions()
                if action not in legal:
                    raise Exception('findPathToClosestDot returned an illegal move: %s!\n%s' %
                            (str(action), str(currentState)))

                currentState = currentState.generateSuccessor(0, action)

        logging.info('Path found with cost %d.' % len(self._actions))

    def findPathToClosestDot(self, gameState):
        """
        Returns a path (a list of actions) to the closest dot, starting from gameState.
        """

        # Here are some useful elements of the startState
        # startPosition = gameState.getPacmanPosition()
        # food = gameState.getFood()
        # walls = gameState.getWalls()
        # problem = AnyFoodSearchProblem(gameState)

        # *** Your Code Here ***
        problem = AnyFoodSearchProblem(gameState)

        return uniformCostSearch(problem)

class AnyFoodSearchProblem(PositionSearchProblem):
    """
    A search problem for finding a path to any food.

    This search problem is just like the PositionSearchProblem,
    but has a different goal test, which you need to fill in below.
    The state space and successor function do not need to be changed.

    The class definition above, `AnyFoodSearchProblem(PositionSearchProblem)`,
    inherits the methods of `pacai.core.search.position.PositionSearchProblem`.

    You can use this search problem to help you fill in
    the `ClosestDotSearchAgent.findPathToClosestDot` method.

    Additional methods to implement:

    `pacai.core.search.position.PositionSearchProblem.isGoal`:
    The state is Pacman's position.
    Fill this in with a goal test that will complete the problem definition.
    """

    def __init__(self, gameState, start = None):
        super().__init__(gameState, goal = None, start = start)

        # Store the food for later reference.
        self.food = gameState.getFood()

    # Goal is reached when target position has some food
    def isGoal(self, gameState):
        return self.food[gameState[0]][gameState[1]]


class ApproximateSearchAgent(BaseAgent):
    """
    Implement your contest entry here.

    Additional methods to implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Get a `pacai.bin.pacman.PacmanGameState`
    and return a `pacai.core.directions.Directions`.

    `pacai.agents.base.BaseAgent.registerInitialState`:
    This method is called before any moves are made.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index)
