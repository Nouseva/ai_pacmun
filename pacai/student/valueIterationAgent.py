from pacai.agents.learning.value import ValueEstimationAgent
from pacai.util import counter

class ValueIterationAgent(ValueEstimationAgent):
    """
    A value iteration agent.

    Make sure to read `pacai.agents.learning` before working on this class.

    A `ValueIterationAgent` takes a `pacai.core.mdp.MarkovDecisionProcess` on initialization,
    and runs value iteration for a given number of iterations using the supplied discount factor.

    Some useful mdp methods you will use:
    `pacai.core.mdp.MarkovDecisionProcess.getStates`,
    `pacai.core.mdp.MarkovDecisionProcess.getPossibleActions`,
    `pacai.core.mdp.MarkovDecisionProcess.getTransitionStatesAndProbs`,
    `pacai.core.mdp.MarkovDecisionProcess.getReward`.

    Additional methods to implement:

    `pacai.agents.learning.value.ValueEstimationAgent.getQValue`:
    The q-value of the state action pair (after the indicated number of value iteration passes).
    Note that value iteration does not necessarily create this quantity,
    and you may have to derive it on the fly.

    `pacai.agents.learning.value.ValueEstimationAgent.getPolicy`:
    The policy is the best action in the given state
    according to the values computed by value iteration.
    You may break ties any way you see fit.
    Note that if there are no legal actions, which is the case at the terminal state,
    you should return None.
    """

    def __init__(self, index, mdp, discountRate = 0.9, iters = 100, **kwargs):
        super().__init__(index)

        self.mdp = mdp
        self.discountRate = discountRate
        self.iters = iters
        self.values = counter.Counter()  # A Counter is a dict with default 0
        self.actions = counter.Counter()

        # print("Starting:", self.mdp.getStates())

        # Compute the values here
        for i in range(self.iters):
            actions_next, values_next = self.valueUpdate(i)
            # No actions left to take
            if actions_next is None:
                break

            self.values = values_next
            self.actions = actions_next

    def getValue(self, state):
        """
        Return the value of the state (computed in __init__).
        """

        return self.values[state]

    def getAction(self, state):
        """
        Returns the policy at the state (no exploration).
        """

        return self.getPolicy(state)

    def getPolicy(self, state):
        action = self.actions[state]
        if action == 0:
            return None
        return action

    def getQValue(self, state, action):
        m = self.mdp
        # action_transitions[] = (next_state, probability)
        action_transitions = m.getTransitionStatesAndProbs(state, action)
        action_value = counter.Counter()
        # print("action_transitions:", action_transitions)
        # action_value = sum(probability * (reward + (discount * prev_value)))
        for n_state, prob in action_transitions:
            # print("State:", state,"n_state:", n_state, "Value:", self.getValue(n_state))
            discount_val = self.discountRate * self.getValue(n_state)
            reward = m.getReward(state, action, n_state)
            # print(discount_val, reward, prob)
            action_value[n_state] = (prob * (reward + discount_val))

        if (len(action_value) == 0):
            return 0
        return action_value.totalCount()

    # Returns an updated value dict, None if there are no possible states
    def valueUpdate(self, depth):
        m = self.mdp

        new_values = counter.Counter()
        new_actions = counter.Counter()
        # Flag that there are some states possible
        some_states = False
        for state in m.getStates():
            if (not some_states):
                some_states = True

            action_values = counter.Counter()
            possible_actions = m.getPossibleActions(state)
            # print("state:", state, "Poss Actions:", possible_actions)
            if (len(possible_actions) == 0):
                continue
            for action in possible_actions:
                action_values[action] = self.getQValue(state, action)

            # print("Action Values:", action_values)
            # Store the value from the maximizing action for this state
            new_actions[state] = action_values.argMax()
            new_values[state] = action_values[new_actions[state]]

        # print("Depth:", depth)
        # print("new_actions:", new_actions)
        # print("new_values:", new_values)
        if (some_states):
            return new_actions, new_values
        return None, None
