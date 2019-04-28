# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        self.qvalues = util.Counter() 

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        stateActionPair = (state, action)
        return self.qvalues[stateActionPair]


    def getBestActionQValuePair(self, state):
        bestAction = None
        bestQValue = 0.0
        
        bestActionsList = []
        possibleActions = self.getLegalActions(state)
        
        #print("Computing best action / Q-value. State = " + str(state) + ", Possible Actions = " + str(possibleActions))
        
        for action in possibleActions:
            actionQValue = self.getQValue(state, action)
            #print("Candidate is: Q-value = " + str(actionQValue) + ", Action = " + str(action))
            
            if not bestActionsList:
                bestActionsList.append(action)
                bestQValue = actionQValue
                #print("  -> Result: Added as first list entry.")
            else:
                if actionQValue > bestQValue:
                    bestActionsList.clear()
                    bestActionsList.append(action)
                    bestQValue = actionQValue
                    #print("  -> Result: New best value. Replaced list and added entry.")
                else:
                    if math.isclose(bestQValue, actionQValue, rel_tol=1e-7):
                        bestActionsList.append(action)
                        #print("  -> Result: Value is tied with best value. Added to list.")
                    #else:
                    #    print("  -> Result: Value is sub-optimal.")
        
        if len(bestActionsList) > 0:
            bestAction = random.choice(bestActionsList) 

        #print("List size for best actions was: " + str(len(bestActionsList)))
        #print("Returning best pair. Q-value = " + str(bestQValue) + ", Action = " + str(bestAction))                
        bestActionQValuePair = (bestAction, bestQValue)
        return bestActionQValuePair

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        bestActionQValuePair = self.getBestActionQValuePair(state)
        return bestActionQValuePair[1]
        

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        bestActionQValuePair = self.getBestActionQValuePair(state)
        return bestActionQValuePair[0]

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        
        if len(legalActions) > 0:
            pickRandomAction = util.flipCoin(self.epsilon)
            if pickRandomAction:
                action = random.choice(legalActions)
            else:
                action = self.computeActionFromQValues(state)

        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        sampleEstimate = reward + self.discount * self.computeValueFromQValues(nextState)
        currentQValue = self.getQValue(state, action)
                
        #newValue = (1 - self.alpha) * currentQValue + self.alpha * sampleEstimate
        newValue = currentQValue + self.alpha * (sampleEstimate - currentQValue)
        
        stateActionPair = (state, action)
        self.qvalues[stateActionPair] = newValue
        
    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        #print("Getting Q-value!!")
        #print("Feature extractor = " + str(self.featExtractor))

        qValue = 0.0
        features = self.featExtractor.getFeatures(state, action)
        #print("Features = " + str(features))
        
        for feature in features.keys():
            featureValue = features[feature]
            #print("Inside loop. Feature = " + str(feature) + ", featureValue = " + str(featureValue))
            
            featureWeight = self.weights[feature]
            qValue += featureValue * featureWeight

        return qValue

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        currentQValue = self.getQValue(state, action)
        
        difference = (reward + self.discount * self.computeValueFromQValues(nextState)) - currentQValue
        
        features = self.featExtractor.getFeatures(state, action)
        for feature in features.keys():
            featureValue = features[feature]
            currentWeight = self.weights[feature]
            newWeight = currentWeight + (self.alpha * difference * featureValue)
            self.weights[feature] = newWeight 

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            print("Current weights: " + str(self.weights))
