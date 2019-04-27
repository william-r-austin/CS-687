# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        iteration = 0
        
        while iteration < self.iterations:
            allStates = self.mdp.getStates()
            newValues = util.Counter()
            
            for state in allStates:
                bestValue = None
                
                possibleActions = self.mdp.getPossibleActions(state)
                
                for action in possibleActions:
                    qValue = self.computeQValueFromValues(state, action)
                    
                    #print("Ugh. Iteration = " + str(iteration) + ", State = " + str(state) + ", Action = " + str(action) + ", Q-value = " + str(qValue))
                    if bestValue is None:
                        bestValue = qValue
                    else:
                        if qValue > bestValue:
                            bestValue = qValue
                
                if bestValue is None:
                    bestValue = 0.0
                
                newValues[state] = bestValue
            
            #print("Done with iteration " + str(iteration) + ", values are: " + str(self.values))
            iteration += 1
            self.values = newValues

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        qValue = 0.0
        transitionStatesAndProbs = self.mdp.getTransitionStatesAndProbs(state, action)
        
        for nextStateAndProb in transitionStatesAndProbs:
            nextState = nextStateAndProb[0]
            nextStateProbability = nextStateAndProb[1]
            
            nextStateReward = self.mdp.getReward(state, action, nextState)
            discountedNextStateValue = self.discount * self.values[nextState]
            
            weightedValue = nextStateProbability * (nextStateReward + discountedNextStateValue)
            
            #print(" Computing Q-value. State = " + str(state) + ", Action = " + str(action) + ", Reward = " + str(nextStateReward) + \
            #      
            #      ", nextStateValue = " + str(self.values[nextState]) + ", Discount = " +  str(self.discount) + ", output = " + str(weightedValue))
            
            qValue += weightedValue
        
        return qValue
        

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        
        actionList = self.mdp.getPossibleActions(state)
        bestActionQValue = None
        bestAction = None
        
        for action in actionList:
            actionQValue = self.computeQValueFromValues(state, action)
            
            if bestAction is None:
                bestAction = action
                bestActionQValue = actionQValue
            else:
                if actionQValue > bestActionQValue:
                    bestAction = action
                    bestActionQValue = actionQValue
        
        return bestAction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        allStates = self.mdp.getStates()
        stateIndex = 0
        
        iteration = 0
        
        while iteration < self.iterations:
            state = allStates[stateIndex]
            
            bestValue = None
            possibleActions = self.mdp.getPossibleActions(state)
            
            for action in possibleActions:
                qValue = self.computeQValueFromValues(state, action)
                
                #print("Ugh. Iteration = " + str(iteration) + ", State = " + str(state) + ", Action = " + str(action) + ", Q-value = " + str(qValue))
                if bestValue is None:
                    bestValue = qValue
                else:
                    if qValue > bestValue:
                        bestValue = qValue
            
            if bestValue is None:
                bestValue = 0.0
            
            self.values[state] = bestValue
            
            iteration += 1
            stateIndex += 1
            if stateIndex >= len(allStates):
                stateIndex = 0
        

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        predecessors = dict()
        
        allStates = self.mdp.getStates()
        
        for state in allStates:
            predecessors[state] = []
        
        for state in allStates:
            possibleActions = self.mdp.getPossibleActions(state)
            
            for action in possibleActions:
                transitionInfoList = self.mdp.getTransitionStatesAndProbs(state, action)
                for transitionInfo in transitionInfoList:
                    nextState = transitionInfo[0]
                    nextStateProbability = transitionInfo[1]
                    
                    if nextStateProbability > 0:
                        if state not in predecessors[nextState]:
                            predecessors[nextState].append(state)

        # Create a new priority queue                    
        priorityQueue = util.PriorityQueue()
        
        for state in allStates:
            print("Got predecessor states for " + str(state) + ", they are: " + str(predecessors[state]))
            
            currentValue = self.values[state]
            
            
        

        
        for state in allStates:
            predecessors[state] = []
