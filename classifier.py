# classifier.py

import random
import numpy as np
from pacman import Directions
import math

class Classifier:
    def __init__(self):
        # Stores the training data from good-moves.txt
        self.data = None
        self.target = None

        # Initialising the tree_classifier attribute to be a RandomForestClassifier with 10 trees and a maximum depth of 7.
        # This choice balances between complexity and performance to avoid overfitting.
        self.tree_classifier = RandomForestClassifier(n_estimators=10, max_depth=7)

    # Method to reset the classifier's data, target, and the random forest classifier itself.
    # Useful for reusing the classifier instance with new training data.
    def reset(self):
        self.data = None
        self.target = None
        # Reinitialising the random forest classifier to its original state.
        self.tree_classifier = RandomForestClassifier(n_estimators=10, max_depth=7)
    
    def fit(self, data, target):
        # Method to fit the classifier with data and target labels. Calls the RandomForest class.
        print("Fitting")
        self.data = data
        self.target = target

        # Fitting the random forest classifier with the provided data and target labels.
        self.tree_classifier.fit(data, target)

    def predict(self, data, legal=None):
        # Method to predict actions based on the input data and ensure the predicted actions are legal.

        # Use the random forest classifier to predict based on the input data.
        predictions = self.tree_classifier.predict(data)
        predicted_moves = [self.convertNumberToMove(pred) for pred in predictions]
        # print('prediction:', predicted_moves)

        # Ensure the predicted moves are legal according to the game's rules.
        legal_moves = [move if move in legal else random.choice(legal) for move in predicted_moves]
        if legal_moves == predicted_moves:
            return predicted_moves
        else:
            # This virtually never gets called, but just catches potential errors.
            print('Final prediction was illegal, so move:', legal_moves)
            return legal_moves

    def convertMoveToNumber(self, move):
        # This is useful for processing directions as categorical data.
        if move == Directions.NORTH:
            return 0
        elif move == Directions.EAST:
            return 1
        elif move == Directions.SOUTH:
            return 2
        elif move == Directions.WEST:
            return 3
        else:
            return None  # Handles any non-directional move like 'Stop' by returning None
        
    def convertNumberToMove(self, number):
        # Opposite transformation to above function
        if number == 0:
            return Directions.NORTH
        elif number == 1:
            return Directions.EAST
        elif number == 2:
            return Directions.SOUTH
        elif number == 3:
            return Directions.WEST



class RandomForestClassifier:
    def __init__(self, n_estimators=10, max_depth=7, max_features=None):
        self.n_estimators = n_estimators  # The number of decision trees in the forest
        self.max_depth = max_depth        # The maximum depth of each tree
        self.max_features = max_features  # The number of features to consider when looking for the best split (not implemented in this version)
        self.trees = []                   # A list to store all the individual trees in the forest
        self.tree_number = 0              # A counter to keep track of the current tree being generated

    def fit(self, X, y):
        # Fit method is used to train the random forest on a given dataset
        self.trees = []
        for _ in range(self.n_estimators):
            self.tree_number += 1

            # Bootstrap sampling: sample with replacement from the dataset at random
            indices = np.random.choice(len(X), len(X))
            bootstrap_X = [X[i] for i in indices] # The sampled feature vectors
            bootstrap_y = [y[i] for i in indices] # The corresponding labels
            
            # Initialise and fit a new tree to the bootstrap sample
            tree = DecisionTreeClassifier(max_depth=self.max_depth, tree_num = self.tree_number)
            tree.fit(bootstrap_X, bootstrap_y)
            self.trees.append(tree) # Add the fitted tree to the list of trees

    def predict(self, X):
        # The predict method is used to make predictions with the random forest
        # print('X:', X)

        # Check if X is a single feature vector or a list of vectors (catches any errors)
        if isinstance(X[0], list):
            # X is already a list of lists (multiple instances)
            is_single_instance = False
        else:
            # X is a single instance; wrap it in a list to standardise structure
            X = [X]
            # print('Edited to make into a list of lists')
            is_single_instance = True

        # Collect predictions from each tree for each instance in X        
        tree_preds = [tree.predict(X) for tree in self.trees]
        print('tree_preds', tree_preds)

        # Aggregate predictions for each instance across all trees
        final_preds = []
        for i in range(len(X)):
            instance_preds = [tree_preds[j][i] for j in range(len(self.trees))]
            final_pred = max(set(instance_preds), key=instance_preds.count)
            final_preds.append(final_pred)

        return final_preds
        


class DecisionTreeClassifier:
    def __init__(self, max_depth = 7, tree_num=0):
        self.root = None  # The root node of the tree, initially None before the tree is built.
        self.max_depth = max_depth  # The maximum allowed depth of the tree to prevent overfitting.
        self.num = 0  # A counter to keep track of the number of nodes in the tree, used for debugging and analysis.
        self.tree_num = tree_num  # An identifier for the tree, useful when using multiple trees in a forest.

    def fit(self, X, y):
        # The fit method is responsible for building the tree using the provided dataset (X, y).
        # Starts the recursive process of building the tree from the root node using the training data.
        self.root = self._build_tree(X, y)

    def predict(self, X):
        # The predict method takes a list of instances (X) and returns predictions for each instance.
        # Ensures that X is in the expected format (a list of instances, each instance being a list of features).
        if not isinstance(X[0], list):
            X = [X]
        # Generates predictions for each instance in X by traversing the tree from the root to the appropriate leaf.
        prediction = [self._predict_single(x, self.root) for x in X]
        return prediction # Returns a list of predictions corresponding to the list of instances.

    # Layer I: Tree Construction

    def _build_tree(self, X, y, depth = 0):
        # Recursively construct the tree from the dataset, starting at the current node.
        # Interesting; using underscore at beginning of function name is convention within classes as it shows private helper not public interface for class
        self.num += 1
        print(f'Forest tree:', self.tree_num, 'Building branch', self.num)

        # Checks for the stopping conditions of the recursion: all targets are the same or maximum depth is reached.
        if len(y) == 1:
            print('Stopping threshold hit because all targets are the same.')
            return DecisionTreeNode(prediction = max(set(y), key=list(y).count))
        elif depth == self.max_depth:
            print('Stopping threshold hit because maximum depth reached.')
            return DecisionTreeNode(prediction = max(set(y), key = list(y).count))
        
        # 1. Select the best question to split the data on, based on the Gini impurity
        best_question = self._select_best_question(X, y)
        if best_question is None:
            print('No valid question found, creating a leaf node.')
            return DecisionTreeNode(prediction=max(set(y), key=list(y).count))
        elif best_question is not None:
            print('Completed, best_question determined as:', best_question)

        # 2. Split dataset absed on the best question found
        true_rows, true_labels, false_rows, false_labels = self._partition(X, y, best_question)

        # 3. Recursively build 'true' and 'false' branch (where question results in 'positive' and 'negative' response, by calling self)
        true_branch = self._build_tree(true_rows, true_labels, depth + 1)
        false_branch = self._build_tree(false_rows, false_labels, depth + 1)
        
        # Returns a new node with the selected question, true branch, and false branch.
        return DecisionTreeNode(question=best_question, true_branch=true_branch, false_branch=false_branch)

    def _select_best_question(self, X, y):
        best_gain = 0                       # Keep track of the best information gain.
        best_question = None                # Keep track of the feature/value that produced it.
        current_uncertainty = self._gini(y) # Using Gini impurity here for highest information gain measure (lowest coefficient)
        n_features = len(X[0])              # number of features

        # Test all the possible splits and return the best question
        for col in range(n_features):  # for each feature
            values = set([row[col] for row in X])  # unique values in the column

            for val in values:  # for each value
                question = (col, val)

                # Try splitting the dataset. Skip the split if it doesn't divide the dataset.
                true_X, true_y, false_X, false_y = self._partition(X, y, question)
                if len(true_X) == 0 or len(false_X) == 0:
                    continue

                # Calculate the information gain from this split
                gain = self._info_gain(true_y, false_y, current_uncertainty)

                if gain >= best_gain:  # if better than current best
                    best_gain, best_question = gain, question

        return best_question

    def _partition(self, X, y, question):
        # Divide dataset into two based on the answer to the question, creating subsets for the true branch and the false branch of the decision tree
        true_rows, false_rows, true_labels, false_labels = [], [], [], []
        feature_index, value = question

        for row, label in zip(X, y):

            if row[feature_index] == value:
                true_rows.append(row)
                true_labels.append(label)

            else:
                false_rows.append(row)
                false_labels.append(label)

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

    # Layer II: Utility Functions
        
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

    def _class_counts(self, y):
        # Counts the number of each type of example in a dataset.
        counts = {}
        for label in y:
            if label not in counts:
                counts[label] = 0
            counts[label] += 1
        return counts

class DecisionTreeNode:
    def __init__(self, question=None, true_branch=None, false_branch=None, prediction=None):
        self.question = question # Condition to split the data with
        self.true_branch = true_branch
        self.false_branch = false_branch
        self.prediction = prediction # Holds class label if its a leaf

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
