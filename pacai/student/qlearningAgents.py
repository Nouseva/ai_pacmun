from pacai.agents.learning.reinforcement import ReinforcementAgent
from pacai.util import reflection
from pacai.util import counter
from pacai.util import probability
import random


class QLearningAgent(ReinforcementAgent):
    """
    A Q-Learning agent.

    Some functions that may be useful:
    `pacai.agents.learning.reinforcement.ReinforcementAgent.getAlpha`:
    Get the learning rate.
    `pacai.agents.learning.reinforcement.ReinforcementAgent.getDiscountRate`:
    `pacai.agents.learning.reinforcement.ReinforcementAgent.getEpsilon`:
    Get the exploration probability.
    `pacai.agents.learning.reinforcement.ReinforcementAgent.getLegalActions`:

    `pacai.util.probability.flipCoin`:
    Flip a coin (get a binary value) with some probability.

    `random.choice`:
    Pick randomly from a list.

    Additional methods to implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Compute the action to take in the current state.
    With probability `pacai.agents.learning.reinforcement.ReinforcementAgent.getEpsilon`,
    we should take a random action and take the best policy action otherwise.
    Note that if there are no legal actions, which is the case at the terminal state,
    you should choose None as the action.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.update`:
    The parent class calls this to observe a state transition and reward.
    You should do your Q-Value update here.
    Note that you should never call this function, it will be called on your behalf.

    DESCRIPTION:
        Stores Q Values in a counter index by (state, action) pairs
        State values and policies are calculated from these stored
        Q values on demand

    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

        # You can initialize Q-values here.
        self.qValues = counter.Counter()

    def getQValue(self, state, action):
        """
        Get the Q-Value for a `pacai.core.gamestate.AbstractGameState`
        and `pacai.core.directions.Directions`.
        Should return 0.0 if the (state, action) pair has never been seen.
        """
        return self.qValues[(state, action)]

    def getValue(self, state):
        """
        Return the value of the best action in a state.
        I.E., the value of the action that solves: `max_action Q(state, action)`.
        Where the max is over legal actions.
        Note that if there are no legal actions, which is the case at the terminal state,
        you should return a value of 0.0.

        This method pairs with `QLearningAgent.getPolicy`,
        which returns the actual best action.
        Whereas this method returns the value of the best action.
        """
        actions = self.getLegalActions(state)
        values = counter.Counter()

        # No legal actions, state is terminal
        if len(actions) == 0:
            return 0.0

        for action in actions:
            values[action] = self.getQValue(state, action)

        actions_sorted = values.sortedKeys()

        return values[actions_sorted[0]]

    def getAction(self, state):
        ideal_action = self.getPolicy(state)
        # Terminal state
        if ideal_action is None:
            return None

        # Check if agent randomly decides to explore
        if probability.flipCoin(self.getEpsilon()):
            action_list = self.getLegalActions(state)
            return random.choice(action_list)

        return ideal_action

    def getPolicy(self, state):
        """
        Return the best action in a state.
        I.E., the action that solves: `max_action Q(state, action)`.
        Where the max is over legal actions.
        Note that if there are no legal actions, which is the case at the terminal state,
        you should return a value of None.

        This method pairs with `QLearningAgent.getValue`,
        which returns the value of the best action.
        Whereas this method returns the best action itself.
        """
        actions = self.getLegalActions(state)
        # No legal actions, state is terminal
        if len(actions) == 0:
            return None

        values = counter.Counter()
        for action in actions:
            values[action] = self.getQValue(state, action)

        actions_sorted = values.sortedKeys()
        max_action = [actions_sorted[0]]
        # print("Actions_sorted", actions_sorted)
        actions_sorted = actions_sorted[1:]
        # print("Max_action", max_action, "Remaining", actions_sorted)

        # Add any other actions that tie for highest
        for action in actions_sorted:
            # Exit loop when actions produce smaller value
            if values[action] < values[max_action[0]]:
                break
            max_action.append(action)

        # Handles both singleton max and tied max actions
        return random.choice(max_action)

    def update(self, state, action, n_state, reward):

        n_action = self.getAction(n_state)
        n_qVal_prev = self.getQValue(n_state, n_action)

        sample = reward + (self.getDiscountRate() * n_qVal_prev)
        qVal_prev = self.qValues[(state, action)]

        self.qValues[(state, action)] = (qVal_prev
                + self.getAlpha() * (sample - qVal_prev))

class PacmanQAgent(QLearningAgent):
    """
    Exactly the same as `QLearningAgent`, but with different default parameters.
    """

    def __init__(self, index, epsilon = 0.05, gamma = 0.8, alpha = 0.2, numTraining = 0, **kwargs):
        kwargs['epsilon'] = epsilon
        kwargs['gamma'] = gamma
        kwargs['alpha'] = alpha
        kwargs['numTraining'] = numTraining

        super().__init__(index, **kwargs)

    def getAction(self, state):
        """
        Simply calls the super getAction method and then informs the parent of an action for Pacman.
        Do not change or remove this method.
        """

        action = super().getAction(state)
        self.doAction(state, action)

        return action

class ApproximateQAgent(PacmanQAgent):
    """
    An approximate Q-learning agent.

    You should only have to overwrite `QLearningAgent.getQValue`
    and `pacai.agents.learning.reinforcement.ReinforcementAgent.update`.
    All other `QLearningAgent` functions should work as is.

    Additional methods to implement:

    `QLearningAgent.getQValue`:
    Should return `Q(state, action) = w * featureVector`,
    where `*` is the dotProduct operator.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.update`:
    Should update your weights based on transition.

    DESCRIPTION: <Write something here so we know what you did.>
    """

    def __init__(self, index,
            extractor = 'pacai.core.featureExtractors.IdentityExtractor', **kwargs):
        super().__init__(index, **kwargs)
        self.featExtractor = reflection.qualifiedImport(extractor)

        # You might want to initialize weights here.
        self.weight = counter.Counter()

    def getQValue(self, state, action):
        feat_vect = self.featExtractor.getFeatures(self, state, action)
        return feat_vect * self.weight

    def update(self, state, action, n_state, reward):
        feat_vect = self.featExtractor.getFeatures(self, state, action)
        n_Val_prev = self.getValue(n_state)

        correction = (reward + (self.getDiscountRate() * n_Val_prev)
                - self.getQValue(state, action))

        # (f_i)/(a * correction)^-1
        for key in feat_vect.keys():
            feat_vect[key] = feat_vect[key] * self.getAlpha() * correction

        self.weight += (feat_vect)

    def final(self, state):
        """
        Called at the end of each game.
        """

        # Call the super-class final method.
        super().final(state)

        # Did we finish training?
        if self.episodesSoFar == self.numTraining:
            # You might want to print your weights here for debugging.
            # *** Your Code Here ***
            pass
