# classifier.py
# Lin Li/26-dec-2021
#
# Use the skeleton below for the classifier and insert your code here.

import random
import numpy as np
from pacman import Directions

import math

class Classifier:
    def __init__(self):
        self.data = None
        self.target = None

        self.tree_classifier = DecisionTreeClassifier(max_depth=7)

    def reset(self):
        self.data = None
        self.target = None

        self.tree_classifier = DecisionTreeClassifier(max_depth=7)
    
    def fit(self, data, target):
        print("Fitting")

        self.data = data
        self.target = target

        # print('Target:', target)
        self.tree_classifier.fit(data, target)

    def convertMoveToNumber(self, move):
        if move == Directions.NORTH:
            return 0
        elif move == Directions.EAST:
            return 1
        elif move == Directions.SOUTH:
            return 2
        elif move == Directions.WEST:
            return 3
        else:
            return None  # For any non-directional move like 'Stop'
        
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
        

    def predict(self, data, legal=None):
        predictions = self.tree_classifier.predict(data)
        # Convert predictions to moves
        predicted_moves = [self.convertNumberToMove(pred) for pred in predictions]
        # Ensure the predictions are legal
        legal_moves = [move if move in legal else random.choice(legal) for move in predicted_moves]
        return legal_moves



    # # Decision Tree classifier
    # # Each branch represents the outcome of the test, each leaf node represents the predicted move
    # def predict(self, data, legal=None):
        
    #     # Each node should split the data into subsets that are as small and pure (as close to having a unitary outcome) as possible
    #     # We can measure the best split with Gini impurity and entropy/information gain
    #     predictions = self.tree_classifier.predict(data)

    #     # Recursively splitting the attribute for each subset until we meet the stopping criteria

    #     # The stopping criteria is gauged depending on the tree. We'll use max_depth 7 based on tests

    #     # Convert numbers back to moves if necessary
    #     # Ensure the predictions are legal; if not, choose randomly from legal
    #     legal_moves = [pred if pred in self.convertMoveToNumber(legal) else random.choice(legal) for pred in predictions]
    #     return legal_moves

class DecisionTreeNode:
    def __init__(self, question=None, true_branch=None, false_branch=None, prediction=None):
        self.question = question # Condition to split the data with
        self.true_branch = true_branch
        self.false_branch = false_branch
        self.prediction = prediction # Holds class label if its a leaf

class DecisionTreeClassifier:
    def __init__(self, max_depth = 7):
        self.root = None
        self.max_depth = max_depth
        self.num = 0

    def fit(self, X, y):
        self.root = self._build_tree(X, y)

    def predict(self, X):
        # Make sure X is iterable
        return [self._predict_single(x, self.root) for x in X]

    # Layer I

    # Recursively construct the tree from the dataset, starting at the current node.
    # Interesting; using underscore at beginning of function name is convention within classes as it shows private helper not public interface for class
    def _build_tree(self, X, y, depth = 0):
        self.num += 1
        print(f'Building tree {self.num}')
        print('len(set(y)):', len(set(y)))

        # Stopping criteria check
        if len(set(y)) == 1 or depth == self.max_depth:
            print('Stopping threshold hit.')
            return DecisionTreeNode(prediction = max(set(y), key=list(y).count))
        
        # 1. Select the best question to split the data on, based on the Gini impurity
        best_question = self._select_best_question(X, y)
        print('best_question:', best_question)

        # 2. Split dataset absed on the best question found
        true_rows, false_rows, true_labels, false_labels = self._partition(X, y, best_question)
        print('Partitioned')

        # 3. Recursively build 'true' branch (where question results in 'positive' response, by calling self
        print('True branch being built')

        print(f'True labels before recursive call: {true_labels}')
        print(f'False labels before recursive call: {false_labels}')
        
        true_branch = self._build_tree(true_rows, true_labels, depth + 1)

        # 4. Recursively build 'false' branch
        print('False branch being built')
        false_branch = self._build_tree(false_rows, false_labels, depth + 1)
        
        return DecisionTreeNode(question=best_question, true_branch=true_branch, false_branch=false_branch)

    # Using Gini impurity here for highest information gain measure (lowest coefficient)
    def _select_best_question(self, X, y):
        best_gain = 0  # Keep track of the best information gain.
        best_question = None  # Keep track of the feature/value that produced it.
        current_uncertainty = self._gini(y)
        n_features = len(X[0])  # number of features

        # Test all the possible splits and return the best question
        for col in range(n_features):  # for each feature
            values = set([row[col] for row in X])  # unique values in the column

            for val in values:  # for each value

                question = (col, val)

                # try splitting the dataset
                true_X, true_y, false_X, false_y = self._partition(X, y, question)
                # Skip the split if it doesn't divide the dataset.
                if len(true_X) == 0 or len(false_X) == 0:
                    continue

                # Calculate the information gain from this split
                gain = self._info_gain(true_y, false_y, current_uncertainty)

                if gain >= best_gain:  # if better than current best
                    best_gain, best_question = gain, question

        return best_question

    # Divide dataset into two based on the answer to the question, creating subsets for the true branch and the false branch of the decision tree
    def _partition(self, X, y, question):
        """Partitions subsets."""

        true_rows, false_rows, true_labels, false_labels = [], [], [], []
        feature_index, value = question

        for row, label in zip(X, y):
            # print(f"For row, label in zip(X,y): Appending label: {label}")  # Debugging line

            if row[feature_index] == value:
                true_rows.append(row)
                true_labels.append(label)
                # Ensure label is a single value, not a list
                # print('true_labels:', true_labels)

            else:
                false_rows.append(row)
                false_labels.append(label)

        print(f"For row, label in zip(X,y): True_label: {true_labels}")  # Debugging line

        return true_rows, true_labels, false_rows, false_labels

    def _predict_single(self, x, node):
        # Base case: we've reached a leaf
        if node.prediction is not None:
            return node.prediction

        # Decide whether to follow the true_branch or the false_branch
        if x[node.question[0]] == node.question[1]:
            return self._predict_single(x, node.true_branch)
        else:
            return self._predict_single(x, node.false_branch)

    # Layer II

    def _gini(self, y):
        counts = self._class_counts(y)
        impurity = 1
        for lbl in counts:
            prob_of_lbl = counts[lbl] / float(len(y))
            impurity -= prob_of_lbl**2
        return impurity

    def _info_gain(self, left, right, current_uncertainty):
        p = float(len(left)) / (len(left) + len(right))
        return current_uncertainty - p * self._gini(left) - (1 - p) * self._gini(right)

    # Layer III

    def _class_counts(self, y):
        """Counts the number of each type of example in a dataset."""
        counts = {}
        for label in y:
            if label not in counts:
                counts[label] = 0
            counts[label] += 1
        return counts
    
# kNN Classifier
# def predict(self, data, legal=None):

#     # Simple 1-NN implementation
#     closest_distance = float('inf')
#     closest_target = None
#     print(f"Legal actions: {legal}")
    
#     for i, data_point in enumerate(self.data):
#         distance = np.sum(np.abs(np.array(data_point) - np.array(data)))
#         if distance < closest_distance:
#             closest_distance = distance
#             closest_target = self.target[i]

#     legal_numbers = [self.convertMoveToNumber(move) for move in legal if self.convertMoveToNumber(move) is not None]
    
#     # Ensure the action is legal; if not, choose randomly from legal actions
#     if closest_target in legal_numbers:
#         print(f"Predicting {closest_target}, which is legal")
#         return closest_target
#     else:
#         random_move = random.choice(legal)
#         # Assuming legal is a list of integers representing legal actions
#         print(f"Predicting {random_move} at random as {closest_target} illegal")
#         return random_move
