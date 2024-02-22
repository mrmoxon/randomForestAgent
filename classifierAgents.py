# classifierAgents.py
# Lin Li/02-Feb-2022
#
# Version 1.0
#
# Some simple agents to work with the PacMan AI projects from:
#
# http://ai.berkeley.edu/
#
# These use a simple API that allow us to control Pacman's interaction with
# the environment adding a layer on top of the AI Berkeley code.
#
# As required by the licensing agreement for the PacMan AI we have:
#
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

# The agents here are extensions written by Simon Parsons, based on the code in
# pacmanAgents.py

from pacman import Directions
from game import Agent
import api
import random
import numpy as np

from classifier import Classifier

def loadData(filepath):
    # open datafile, extract content into an array, and close.
    with open(filepath, 'r') as f: lines = f.readlines()
    # Now extract data, which is in the form of strings, into an
    # array of numbers, and separate into matched data and target
    # variables.
    data, target = [], []
    for line in lines:
        line = [int(c) for c in line if c>='0' and c<='9']
        data.append(line[:-1])
        target.append(line[-1])
    return data, target

# ClassifierAgent
#
# An agent that runs a classifier to decide what to do.
class ClassifierAgent(Agent):

    # Constructor. This gets run when the agent starts up.
    def __init__(self):
        print("Initialising")
                
    # This gets run on startup. Has access to state information.
    # Here we use it to load the training data.
    def registerInitialState(self, state):

        self.data, self.target = loadData('good-moves.txt')
        
        # data and target are both arrays of arbitrary length.
        # Data is an array of arrays of integers (0 or 1) indicating state.
        # Target is an array of integers 0-3 indicating the action (N, E, S, W), taken in that state.            
        self.classifier = Classifier()

        # Fit your model to the data from 'good-moves', so in each state, have a route.
        self.classifier.fit(self.data, self.target)
        
    # Tidy up when Pacman dies
    def final(self, state):
        print("I'm done!")        
        self.classifier.reset()
        
    # Turn the numbers from the feature set into actions:
    def convertNumberToMove(self, number):
        if number == 0:
            return Directions.NORTH
        elif number == 1:
            return Directions.EAST
        elif number == 2:
            return Directions.SOUTH
        elif number == 3:
            return Directions.WEST

    # Here we just run the classifier to decide what to do, based on the model we've fitted.
    def getAction(self, state):

        # How we access the features.
        features = api.getFeatureVector(state)        
        # Get the actions we can try.
        legal = api.legalActions(state)
        
        # predict what action to take
        action = self.convertNumberToMove(self.classifier.predict(features, legal))
        # randomly pick a legal action if the estimated action is illegal
        action = action if action in legal else random.choice(legal)
        
        # getAction has to return a move. We need to pass the set of legal
        # moves to the API so it can do some safety checking.
        return api.makeMove(action, legal)

