# Modified version of random-forests's orginal which can be found at: https://github.com/random-forests/tutorials/blob/master/decision_tree.py
# Tutorial at https://www.youtube.com/watch?v=LDRbO9a6XPU

# For Python 2 / 3 compatability
from __future__ import print_function

class Leaf:
    """A Leaf node classifies data.
    This holds a dictionary of class (e.g., "Apple") -> number of times it appears in the rows from the training data that reach this leaf."""
    
    def __init__(self, rows):
        self.predictions = self.class_counts(rows)

    # Counts the number of each type of example in a dataset.
    def class_counts(self, rows):
        counts = {}  # a dictionary of label -> count.
        for row in rows:
            # in our dataset format, the label is always the last column
            label = row[-1]
            if label not in counts:
                counts[label] = 0
            counts[label] += 1
        return counts

class Decision_Node:
    """A Decision Node asks a question.
    This holds a reference to the question, and to the two child nodes.
    """
    def __init__(self, question, true_branch, false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch

class Question:
    """A Question is used to partition a dataset.
    This class just records a 'column number' (e.g., 0 for Color) and a 'column value' (e.g., Green). The 'match' method is used to compare
    the feature value in an example to the feature value stored in the question."""
    def __init__(self, column, value):
        self.column = column
        self.value = value

    # Compares the feature value in an example to the feature value in this question.
    def match(self, example):
        val = example[self.column]
        if self.is_numeric(val):
            return val >= self.value
        else:
            return val == self.value

    # Checks if a value is numeric.
    def is_numeric(self, value):
        return isinstance(value, int) or isinstance(value, float)


    # Helper method to print the question in a readable format.
    def __repr__(self):
        condition = "=="
        if self.is_numeric(self.value):
            condition = ">="
        return "Is %s %s?" % (condition, str(self.value))

# Toy dataset.
# Format: each row is an example.
# The last column is the label.
# The first two columns are features.
class Decision_Tree:
    training_data = []

    def __init__(self, training_data):
        self.training_data = training_data
        self.tree = self.build_tree(self.training_data)

    # Finds the unique values for a column in a dataset.
    def unique_vals(self, rows, col):
        return set([row[col] for row in rows])

    # Counts the number of each type of example in a dataset.
    def class_counts(self, rows):
        counts = {}  # a dictionary of label -> count.
        for row in rows:
            # in our dataset format, the label is always the last column
            label = row[-1]
            if label not in counts:
                counts[label] = 0
            counts[label] += 1
        return counts

    def partition(self, rows, question):
        """Partitions a dataset.
        For each row in the dataset, check if it matches the question. 
        If so, add it to 'true rows', otherwise, add it to 'false rows'."""
        true_rows, false_rows = [], []
        for row in rows:
            if question.match(row):
                true_rows.append(row)
            else:
                false_rows.append(row)
        return true_rows, false_rows


    # Calculates the Gini Impurity for a list of rows.
    def gini(self, rows):
        """There are a few different ways to do this, I thought this one was
        the most concise. See: https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity """
        counts = self.class_counts(rows)
        impurity = 1
        for lbl in counts:
            prob_of_lbl = counts[lbl] / float(len(rows))
            impurity -= prob_of_lbl**2
        return impurity

    #######
    # Demo:
    # Let's look at some example to understand how Gini Impurity works.
    #
    # First, we'll look at a dataset with no mixing.
    # no_mixing = [['Apple'],
    #              ['Apple']]
    # this will return 0
    # gini(no_mixing)
    #
    # Now, we'll look at dataset with a 50:50 apples:oranges ratio
    # some_mixing = [['Apple'],
    #               ['Orange']]
    # this will return 0.5 - meaning, there's a 50% chance of misclassifying
    # a random example we draw from the dataset.
    # gini(some_mixing)
    #
    # Now, we'll look at a dataset with many different labels
    # lots_of_mixing = [['Apple'],
    #                  ['Orange'],
    #                  ['Grape'],
    #                  ['Grapefruit'],
    #                  ['Blueberry']]
    # This will return 0.8
    # gini(lots_of_mixing)
    #######

    def info_gain(self, left, right, current_uncertainty):
        """Information Gain.
        The uncertainty of the starting node, minus the weighted impurity of
        two child nodes.
        """
        p = float(len(left)) / (len(left) + len(right))
        return current_uncertainty - p * self.gini(left) - (1 - p) * self.gini(right)

    #######
    # Demo:
    # Calculate the uncertainy of our training data.
    # current_uncertainty = gini(training_data)
    #
    # How much information do we gain by partioning on 'Green'?
    # true_rows, false_rows = partition(training_data, Question(0, 'Green'))
    # info_gain(true_rows, false_rows, current_uncertainty)
    #
    # What about if we partioned on 'Red' instead?
    # true_rows, false_rows = partition(training_data, Question(0,'Red'))
    # info_gain(true_rows, false_rows, current_uncertainty)
    #
    # It looks like we learned more using 'Red' (0.37), than 'Green' (0.14).
    # Why? Look at the different splits that result, and see which one
    # looks more 'unmixed' to you.
    # true_rows, false_rows = partition(training_data, Question(0,'Red'))
    #
    # Here, the true_rows contain only 'Grapes'.
    # true_rows
    #
    # And the false rows contain two types of fruit. Not too bad.
    # false_rows
    #
    # On the other hand, partitioning by Green doesn't help so much.
    # true_rows, false_rows = partition(training_data, Question(0,'Green'))
    #
    # We've isolated one apple in the true rows.
    # true_rows
    #
    # But, the false-rows are badly mixed up.
    # false_rows
    #######

    # Finds the best question to ask by iterating over every feature / value and calculating the information gain.
    def find_best_split(self, rows):
        best_gain = 0  # keep track of the best information gain
        best_question = None  # keep train of the feature / value that produced it
        current_uncertainty = self.gini(rows)
        n_features = len(rows[0]) - 1  # number of columns

        for col in range(n_features):  # for each feature

            values = set([row[col] for row in rows])  # unique values in the column

            for val in values:  # for each value

                question = Question(col, val)

                # try splitting the dataset
                true_rows, false_rows = self.partition(rows, question)

                # Skip this split if it doesn't divide the dataset.
                #if len(true_rows) == 0 or len(false_rows) == 0:
                #    continue

                if len(true_rows) > 0 and len(false_rows) > 0:
                    gain = self.info_gain(true_rows, false_rows, current_uncertainty) # Calculate the information gain from this split
                    if gain > best_gain: # You can use either '>' or '>=' here
                        best_gain, best_question = gain, question

        return best_gain, best_question

    def build_tree(self, rows):
        """Builds the tree.
        Rules of recursion: 1) Believe that it works. 2) Start by checking
        for the base case (no further information gain). 3) Prepare for
        giant stack traces.
        """

        # Try partitioing the dataset on each of the unique attribute,
        # calculate the information gain,
        # and return the question that produces the highest gain.
        gain, question = self.find_best_split(rows)

        # Base case: no further info gain
        # Since we can ask no further questions,
        # we'll return a leaf.
        if gain == 0:
            return Leaf(rows)

        # If we reach here, we have found a useful feature / value
        # to partition on.
        true_rows, false_rows = self.partition(rows, question)

        # Recursively build the true branch.
        true_branch = self.build_tree(true_rows)

        # Recursively build the false branch.
        false_branch = self.build_tree(false_rows)

        # Return a Question node.
        # This records the best feature / value to ask at this point,
        # as well as the branches to follow
        # dependingo on the answer.
        return Decision_Node(question, true_branch, false_branch)

    # def print_tree(self, *args)
    #     print(len(args))
    #     print_tree(self.tree)
    def print_tree(self):
        self.print_nodes(self.tree)

    def print_nodes(self, node, spacing=""):
        """World's most elegant tree printing function."""

        # Base case: we've reached a leaf
        if isinstance(node, Leaf):
            print (spacing + "Predict", node.predictions)
            return

        # Print the question at this node
        print (spacing + str(node.question))

        # Call this function recursively on the true branch
        print (spacing + '--> True:')
        self.print_nodes(node.true_branch, spacing + "  ")

        # Call this function recursively on the false branch
        print (spacing + '--> False:')
        self.print_nodes(node.false_branch, spacing + "  ")


    def classify(self, row, node):
        """See the 'rules of recursion' above."""

        # Base case: we've reached a leaf
        if isinstance(node, Leaf):
            return node.predictions

        # Decide whether to follow the true-branch or the false-branch.
        # Compare the feature / value stored in the node,
        # to the example we're considering.
        if node.question.match(row):
            return self.classify(row, node.true_branch)
        else:
            return self.classify(row, node.false_branch)

    def print_leaf(self, counts):
        """A nicer way to print the predictions at a leaf."""
        total = sum(counts.values()) * 1.0
        probs = {}
        for lbl in counts.keys():
            probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
        return probs

    def test(self, decision_tree, rows):
        for row in rows:
            print ("Actual: %s. Predicted: %s" % (row[-1], self.print_leaf(self.classify(row, decision_tree))))

    def predict(self, decision_tree, row):
            return self.print_leaf(self.classify(row, decision_tree))

    def binary_prediction(self, decision_tree, row):
        
        counts = self.classify(row, decision_tree)
        if -1 not in counts:
            return 1
        if 1 not in counts:
            return -1

        if counts[-1] > counts[1]:
            return -1
        elif counts[-1] > counts[1]:
            return 1
        else:
            print("ERROR")
            return            


if __name__ == '__main__':

    training_data = [
        ['Green', 3, 'grapf', 'Apple'],
        ['Yellow', 3, 'apf', 'Apple'],
        ['Red', 1, 'grapf', 'Grape'],
        ['Red', 1, 'grapf','Grape'],
        ['Yellow', 3, 'lepf', 'Lemon']
    ]

    my_tree = Decision_Tree(training_data)

    my_tree.print_tree()

    # Evaluate
    testing_data = [
        ['Green', 3, 'apf', 'Apple'],
        ['Yellow', 4, 'grapf', 'Apple'],
        ['Red', 2, 'grapf', 'Grape'],
        ['Red', 1, 'apf','Grape'],
        ['Yellow', 3, 'lepf', 'Lemon'],
    ]

    for row in testing_data:
        print ("Actual: %s. Predicted: %s" %
               (row[-1], my_tree.print_leaf(my_tree.classify(row, my_tree.tree))))

# Next steps
# - add support for missing (or unseen) attributes
# - prune the tree to prevent overfitting