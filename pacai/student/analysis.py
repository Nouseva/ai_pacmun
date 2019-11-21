"""
Analysis question.
Change these default values to obtain the specified policies through value iteration.
If any question is not possible, return just the constant NOT_POSSIBLE:
```
return NOT_POSSIBLE
```
"""

NOT_POSSIBLE = None

def question2():
    """
    By reducing the noise, the agent becomes less
    concerned with 'falling off the bridge' so is
    willing to cross
    """

    answerDiscount = 0.9
    answerNoise = 0

    return answerDiscount, answerNoise

def question3a():
    """
    Lower the noise to prefer the red path
    Lower discount to prefer more immediate solutions
    """

    answerDiscount = 0.3
    answerNoise = 0
    answerLivingReward = 0.0

    return answerDiscount, answerNoise, answerLivingReward

def question3b():
    """
    Increase noise to prefer green path
    Change discount closer to zero to prefer immediate solution
    Decrease living reward to avoid safely bumping into walls
    """

    answerDiscount = 0.4
    answerNoise = 0.3
    answerLivingReward = -0.2

    return answerDiscount, answerNoise, answerLivingReward

def question3c():
    """
    Decrease noise to reduce concern for 'dangerous' paths
    Decrease living reward to move swiftly to an exit
    Leave discount close to one to prefer longer term solution

    """

    answerDiscount = 0.9
    answerNoise = 0
    answerLivingReward = -0.2

    return answerDiscount, answerNoise, answerLivingReward

def question3d():
    """
    Increase noise to prefer safer path
    Leave discount close to one to prefer longer term solution
    Decrease living reward to move swiftly to a terminal
    """

    answerDiscount = 0.9
    answerNoise = 0.4
    answerLivingReward = -0.1

    return answerDiscount, answerNoise, answerLivingReward

def question3e():
    """
    Increase living reward to positive
    Eliminate discount by setting to one
    Both of which cause the agent to try and survive as long as possible
    Increase noise to move away from edge and terminal states
    """

    answerDiscount = 1
    answerNoise = 0.4
    answerLivingReward = 0.3

    return answerDiscount, answerNoise, answerLivingReward

def question6():
    """
    [Enter a description of what you did here.]
    """

    answerEpsilon = 0.3
    answerLearningRate = 0.5

    return answerEpsilon, answerLearningRate

if __name__ == '__main__':
    questions = [
        question2,
        question3a,
        question3b,
        question3c,
        question3d,
        question3e,
        question6,
    ]

    print('Answers to analysis questions:')
    for question in questions:
        response = question()
        print('    Question %-10s:\t%s' % (question.__name__, str(response)))
