import random

from pacai.agents.base import BaseAgent
# from pacai.agents.ghost.random import RandomGhost
from pacai.agents.search.multiagent import MultiAgentSearchAgent
from pacai.core import distance
# from pacai.core.actions import Actions
from pacai.core.directions import Directions
from pacai.core.distanceCalculator import Distancer
from pacai.student.search import aStarSearch
from pacai.student import searchAgents
from pacai.util.counter import Counter

class ReflexAgent(BaseAgent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.
    You are welcome to change it in any way you see fit,
    so long as you don't touch the method headers.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index)

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        `ReflexAgent.getAction` chooses among the best options according to the evaluation function.

        Just like in the previous project, this method takes a
        `pacai.core.gamestate.AbstractGameState` and returns some value from
        `pacai.core.directions.Directions`.
        """

        # Collect legal moves.
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions.
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best.

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current `pacai.bin.pacman.PacmanGameState`
        and an action, and returns a number, where higher numbers are better.
        Make sure to understand the range of different values before you combine them
        in your evaluation function.
        """

        # Values can be modified to change importance of certain elements
        VALUE_GHOST = 2
        VALUE_FOOD = 1

        # Distance at which ghosts matter to evaluation
        GHOST_DISTANCE = 6

        successorGameState = currentGameState.generatePacmanSuccessor(action)

        pacman_pos_n = successorGameState.getPacmanPosition()

        # Useful information you can extract.
        # newPosition = successorGameState.getPacmanPosition()
        # oldFood = currentGameState.getFood()
        # newGhostStates = successorGameState.getGhostStates()
        # newScaredTimes = [ghostState.getScaredTimer() for ghostState in newGhostStates]

        new_ghost_states = successorGameState.getGhostStates()
        new_ghost_info = [(ghost_state.getScaredTimer(), ghost_state.getPosition())
                for ghost_state in new_ghost_states]

        food_left_n = successorGameState.getFood().asList()
        # Base food score is dependent on eaten food
        if (len(food_left_n) == 0):
            food_score = VALUE_FOOD
        else:
            food_dist_n = min([distance.euclidean(food, pacman_pos_n)
                for food in food_left_n])

            food_score = VALUE_FOOD / food_dist_n

        ghost_score = 0
        for ghost in new_ghost_info:
            ghost_dist = distance.euclidean(ghost[1], pacman_pos_n)
            # Ghost will be too far to matter
            if (ghost_dist > GHOST_DISTANCE):
                continue
            # Ghost will collide with pacman
            elif (ghost_dist == 0):
                # Pacman has been eaten
                if (ghost[0] == 0):
                    return -VALUE_GHOST
                # Pacman has eaten
                else:
                    ghost_score = ghost_score + VALUE_GHOST

            # Ghost is in range to be wary of
            else:
                # Ghost is brave, try to avoid
                if (ghost[0] == 0):
                    ghost_score = ghost_score - (VALUE_GHOST / ghost_dist)
                # Ghost is scared, try to eat
                else:
                    ghost_score = ghost_score + (VALUE_GHOST / ghost_dist)

        return successorGameState.getScore() + ghost_score + food_score

class MinimaxAgent(MultiAgentSearchAgent):
    """
    A minimax agent.

    Here are some method calls that might be useful when implementing minimax.

    `pacai.core.gamestate.AbstractGameState.getNumAgents()`:
    Get the total number of agents in the game

    `pacai.core.gamestate.AbstractGameState.getLegalActions`:
    Returns a list of legal actions for an agent.
    Pacman is always at index 0, and ghosts are >= 1.

    `pacai.core.gamestate.AbstractGameState.generateSuccessor`:
    Get the successor game state after an agent takes an action.

    `pacai.core.directions.Directions.STOP`:
    The stop direction, which is always legal, but you may not want to include in your search.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    # kwargs: [evalFn, depth, kwargs]
    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    # Returns the action an agent to maximize result
    def getAction(self, gameState):
        # Call recursive maximizing function
        agentAction = self.getMax(gameState, 0, 0)

        # print("Found:", agentAction)
        return agentAction

    # Returns tuple of (minimal value of successorStates, action to reach state)
    def getMin(self, gameState, agentIndex, depth):
        # Strip the action from the state
        if (type(gameState) == tuple):
            gameState = gameState[0]
        currentAgentActions = gameState.getLegalActions(agentIndex)

        # Current agent has no actions available, game must have ended
        if (len(currentAgentActions) == 0):
            return (self.getEvaluationFunction()(gameState))

        # possibleStates is tuple (gameState, action to reach gameState)
        possibleStates = [(gameState.generateSuccessor(agentIndex, action))
                for action in currentAgentActions]

        # There is no further agents to check
        if (depth == self.getTreeDepth()) and (agentIndex == gameState.getNumAgents() - 1):
            # actionVals is tuple (value of state, action to reach state)
            actionVals = [(self.getEvaluationFunction()(state))
                    for state in possibleStates]
            return min(actionVals)

        nextAgent = (agentIndex + 1) % gameState.getNumAgents()
        actionVals = None

        # The next agent is not pacman (not MAX)
        if (nextAgent > 0):
            actionVals = [self.getMin(state, nextAgent, depth)
                    for state in possibleStates]
        else:
            actionVals = [self.getMax(state, nextAgent, depth + 1)
                    for state in possibleStates]

        return min(actionVals)

    # Returns tuple of (maximal value of successorStates, action to reach state)
    def getMax(self, gameState, agentIndex, depth):
        # Strip the action from the state
        if (type(gameState) == tuple):
            gameState = gameState[0]

        # Get actions for pacman, excluding the no movement option
        currentAgentActions = gameState.getLegalActions(agentIndex)
        if (Directions.STOP in currentAgentActions):
            currentAgentActions.remove(Directions.STOP)

        # Pacman has no survivable moves
        if len(currentAgentActions) == 0:
            return (self.getEvaluationFunction()(gameState))

        possibleStates = [(gameState.generateSuccessor(agentIndex, action))
                for action in currentAgentActions]

        nextAgent = (agentIndex + 1) % gameState.getNumAgents()

        if (nextAgent > 0):
            actionVals = [self.getMin(state, nextAgent, depth)
                    for state in possibleStates]

            # If it is the first call to getMax, return the maximizing action
            if (depth == 0):
                return currentAgentActions[actionVals.index(max(actionVals))]
            # Any other call, return the max value
            else:
                return max(actionVals)
        else:
            print("Error: No advasarial Agent available")


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    A minimax agent with alpha-beta pruning.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index)

    # Returns the action an agent to maximize result
    def getAction(self, gameState):

        # Call recursive maximizing function, w/ pruning
        agentAction = self.getMax(gameState, 0, 0, -9999999, 9999999)

        # print("Found:", agentAction)
        return agentAction

    # Returns tuple of (min val, alpha, beta)
    def getMin(self, gameState, agentIndex, depth, alpha, beta):
        currentAgentActions = gameState.getLegalActions(agentIndex)

        # Current agent has no actions available, game must have ended
        if (len(currentAgentActions) == 0):
            return (self.getEvaluationFunction()(gameState))

        # possibleStates is tuple (gameState, action to reach gameState)
        possibleStates = [(gameState.generateSuccessor(agentIndex, action))
                for action in currentAgentActions]

        # There is no further agents to check
        if (depth == self.getTreeDepth()) and (agentIndex == gameState.getNumAgents() - 1):
            # actionVals is tuple (value of state, action to reach state)
            actionVals = [(self.getEvaluationFunction()(state))
                    for state in possibleStates]
            return (min(actionVals))

        nextAgent = (agentIndex + 1) % gameState.getNumAgents()
        nextDepth = None

        val = 9999999
        valueRetrivalFunction = None
        # The next agent is not pacman (not MAX)
        if (nextAgent > 0):
            nextDepth = depth
            valueRetrivalFunction = self.getMin
        # The next agent is pacman (MAX)
        else:
            nextDepth = depth + 1
            valueRetrivalFunction = self.getMax

            # The preceeding agent was pacman
        if ((agentIndex - 1) == 0):

            for state in possibleStates:
                v = valueRetrivalFunction(state, nextAgent, nextDepth, alpha, beta)
                val = min(val, v)
                # Value found does not exceed alpha, MAX agent will prefer alpha
                if (val <= alpha):
                    return (val)
                beta = min(beta, val)
            return (val)

            # The preceeding agent was another ghost, so all paths must be searched
        else:
            actionVals = [valueRetrivalFunction(state, nextAgent, nextDepth, alpha, beta)
                    for state in possibleStates]

        # Handles prev_ghost->current_ghost->some_agent transitions
        # print("Depth:", depth, "Agent:", agentIndex, actionVals)
        return (min(actionVals))

    # Returns either: maximizing action, or tuple (max val, alpha, beta)
    def getMax(self, gameState, agentIndex, depth, alpha, beta):

        # Get actions for pacman, excluding the no movement option
        currentAgentActions = gameState.getLegalActions(agentIndex)
        if Directions.STOP in currentAgentActions:
            currentAgentActions.remove(Directions.STOP)

        # Pacman has no survivable moves
        if len(currentAgentActions) == 0:
            return (self.getEvaluationFunction()(gameState))

        possibleStates = [(gameState.generateSuccessor(agentIndex, action))
                for action in currentAgentActions]

        nextAgent = (agentIndex + 1) % gameState.getNumAgents()
        stateScores = []

        if (nextAgent > 0):
            val = -9999999
            for state in possibleStates:
                v = self.getMin(state, nextAgent, depth, alpha, beta)
                # print(v, alpha, beta)
                stateScores.append(v)

                val = max(val, v)
                if (val >= beta):
                    break
                # if (val >= beta) return (val, alpha, beta)
                alpha = max(alpha, val)

            # If it is the first call to getMax, return the maximizing action
            if (depth == 0):
                # print("Found action val:", val, "stateScores:", stateScores)
                return currentAgentActions[stateScores.index(val)]
            # Any other call, return the max value
            else:
                return (val)
        else:
            print("Error: No advasarial Agent available")

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    An expectimax agent.

    All ghosts should be modeled as choosing uniformly at random from their legal moves.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the expectimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    # Returns the action an agent to maximize result
    def getAction(self, gameState):

        # Call recursive maximizing function, w/o pruning
        agentAction = self.getMax(gameState, 0, 0)

        # print("Found:", agentAction)
        return agentAction

    def getExpected(self, gameState, agentIndex, depth):
        currentAgentActions = gameState.getLegalActions(agentIndex)

        # Current agent has no actions available, game ended
        if (len(currentAgentActions) == 0):
            return self.getEvaluationFunction()(gameState)

        # TODO: Use RandomGhost to determine distribution instead of building from scratch
        stateDistribution = Counter()
        for a in currentAgentActions:
            stateDistribution[a] = 1.0
        stateDistribution.normalize()
        # print(stateDistribution)

        possibleStates = [(gameState.generateSuccessor(agentIndex, action))
                for action in currentAgentActions]

        if (depth == self.getTreeDepth()) and (agentIndex == gameState.getNumAgents() - 1):
            actionVals = [(self.getEvaluationFunction()(possibleStates[i])
                * stateDistribution[currentAgentActions[i]])
                for i in range(len(possibleStates))]

            # print("Depth:", depth, "Agent:", agentIndex, actionVals)
            return (sum(actionVals))

        nextAgent = (agentIndex + 1) % gameState.getNumAgents()
        nextDepth = None

        valueRetrivalFunction = None
        # The next agent is not pacman
        if (nextAgent > 0):
            nextDepth = depth
            valueRetrivalFunction = self.getExpected
        # The next agent is pacman (MAX)
        else:
            nextDepth = depth + 1
            valueRetrivalFunction = self.getMax

        # The values are multiplied by the expetation
        actionVals = [(valueRetrivalFunction(possibleStates[i], nextAgent, nextDepth)
            * stateDistribution[currentAgentActions[i]])
            for i in range(len(possibleStates))]

        # Handles prev_ghost->current_ghost->some_agent transitions
        # print("Depth:", depth, "Agent:", agentIndex, actionVals)
        return (sum(actionVals))

    # Returns either: maximizing action, or maximal value
    def getMax(self, gameState, agentIndex, depth):

        # Get actions for pacman, excluding the no movement option
        currentAgentActions = gameState.getLegalActions(agentIndex)
        if Directions.STOP in currentAgentActions:
            currentAgentActions.remove(Directions.STOP)

        # Pacman has no survivable moves
        if len(currentAgentActions) == 0:
            return (self.getEvaluationFunction()(gameState))

        possibleStates = [(gameState.generateSuccessor(agentIndex, action))
                for action in currentAgentActions]

        nextAgent = (agentIndex + 1) % gameState.getNumAgents()
        stateScores = None

        if (nextAgent > 0):
            stateScores = [(self.getExpected(state, nextAgent, depth))
                    for state in possibleStates]

            # If it is the first call to getMax, return the maximizing action
            if (depth == 0):
                # print("Found action val:", val, "stateScores:", stateScores)
                return currentAgentActions[stateScores.index(max(stateScores))]
            # Any other call, return the max value
            else:
                return max(stateScores)
        else:
            print("Error: No advasarial Agent available")

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable evaluation function.

    DESCRIPTION:
        - When state results in a completed game, the score for the state is returned
        - Ghosts that are deemed too far are not considered for evaluation
            - Two passes to deterimine distance:
                - Manhattan gives rough distance from pacman
                - Distancer is used to get the exact distance, only used when within strike range
        - Target food is determined using a* on the AnyFoodSearchProblem
    """
    # If game has ended, return the score and nothing else
    if (currentGameState.isOver()):
        return currentGameState.getScore()

    # Values can be modified to change importance of certain elements
    VALUE_GHOST = 50
    VALUE_FOOD = 6

    # Distance at which ghosts matter to evaluation
    GHOST_DISTANCE = 5

    score_food = 0
    score_ghost = 0

    distancer = Distancer(currentGameState.getInitialLayout())

    state_pacman = currentGameState.getPacmanState()
    state_ghost = currentGameState.getGhostStates()

    posi_pacman = state_pacman.getPosition()
    # Rough distances between pacman and ghosts
    dist_ghost = [distance.manhattan(ghost.getPosition(), posi_pacman)
            for ghost in state_ghost]

    # For ghosts that appear close, calculate the actual distance to them
    for i in range(len(dist_ghost)):
        # if (dist_ghost[i] < GHOST_DISTANCE):
        dist_ghost[i] = distancer.getDistance(state_ghost[i].getPosition(), posi_pacman)

    for i in range(len(state_ghost)):
        # Ghost will be too far to matter
        if (dist_ghost[i] > GHOST_DISTANCE):
            continue
        # Ghost is in range to be wary of
        else:
            # Ghost is brave, avoid
            if (state_ghost[i].isBraveGhost()):
                score_ghost = -(VALUE_GHOST / dist_ghost[i])
            # Ghost is scared
            else:
                # Ghost will become brave before reaching, ignore
                if (state_ghost[i].getScaredTimer() < dist_ghost[i]):
                    continue
                # Ghost will remain scared when reached if pursued
                else:
                    score_ghost = (VALUE_GHOST / dist_ghost[i])

    if (currentGameState.getNumFood() > 0):
        # Check distance to closest food
        prob_food = searchAgents.AnyFoodSearchProblem(currentGameState)
        steps_food = aStarSearch(prob_food, expec_food_heuristic)

        score_food = VALUE_FOOD / len(steps_food)
    else:
        print("Error: Game has no pellets to target")

    total = score_ghost + score_food + currentGameState.getScore()
    return total

def expec_food_heuristic(state, problem):
    if (len(problem.food.asList()) == 0):
        return 0

    pellet_list = problem.food.asList()
    dist = set()
    for p in pellet_list:
        dist.add(distance.manhattan(p, state))

    return min(dist)

class ContestAgent(MultiAgentSearchAgent):
    """
    Your agent for the mini-contest.

    You can use any method you want and search to any depth you want.
    Just remember that the mini-contest is timed, so you have to trade off speed and computation.

    Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
    just make a beeline straight towards Pacman (or away if they're scared!)

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`
    """

    def __init__(self, index, **kwargs):
        super().__init__(index)
