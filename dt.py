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


    def info_gain(self, left, right, current_uncertainty):
        """Information Gain.
        The uncertainty of the starting node, minus the weighted impurity of
        two child nodes.
        """
        p = float(len(left)) / (len(left) + len(right))
        return current_uncertainty - p * self.gini(left) - (1 - p) * self.gini(right)

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

        # TODO: Ena krockodimunnen fel, för trött för att fixa nu
        if counts[-1] > counts[1]:
            return -1
        elif counts[-1] <= counts[1]:
            return 1
        else:
            print("ERROR")
            return            


if __name__ == '__main__':
    import data_sort
    import time

    start_time = time.time()
    number_of_features = 7
    training_data = data_sort.makeSet("datasets/car.txt", number_of_features)
    # training_data = data_sort.binaryfy(training_data)
    test_data = data_sort.makeSet("datasets/car_test.txt", number_of_features)
    # test_data = data_sort.binaryfy(test_data)
    print("Training")

    accuracy = 0
    correct_prediction = 0
    true_positive_minus = 0
    false_positive_minus = 0
    false_negative_minus = 0
    true_positive_plus = 0
    false_negative_plus = 0
    false_positive_plus = 0
    my_tree = Decision_Tree(training_data)

    # my_tree.print_tree()
    print("Running test")
    successful_prediction = 0
    for i in range(0, len(test_data)):
        correct_lable = test_data[i][-1]
        predicted_lable = list(my_tree.predict(my_tree.tree, test_data[i]).keys())[0]
        # print("Pred", list(my_tree.predict(my_tree.tree, test_data[i]).keys())[0])
        # print("Corr", correct_lable)
        if predicted_lable == correct_lable:
            successful_prediction += 1
            if "acc" == predicted_lable:
                true_positive_plus += 1
            else:
                true_positive_minus += 1
        else:
            if "acc" == predicted_lable:
                
                false_positive_plus += 1
                false_negative_minus += 1
            else:
                false_negative_plus += 1
                false_positive_minus += 1

    precission_plus = 0 if true_positive_plus == 0 else true_positive_plus / (true_positive_plus + false_positive_plus)
    recall_plus = 0 if true_positive_plus == 0 else true_positive_plus / (true_positive_plus + false_negative_plus)
    precission_minus = 0 if true_positive_minus == 0 else true_positive_minus/(true_positive_minus + false_positive_minus)
    recall_minus = 0 if true_positive_minus == 0 else true_positive_minus/(true_positive_minus+false_negative_minus)
    F1 = 2*((precission_plus+precission_minus)/2 * (recall_plus+recall_minus)/2) / ((precission_plus+precission_minus)/2 + (recall_plus+recall_minus)/2)
    accuracy = successful_prediction/len(test_data)

    print("Time", str(time.time()-start_time))
    print("Accuracy: ", str(accuracy))
    print("Precision: ", str(precission_plus))
    print("Recall: ", str(recall_plus))
    print("F1 score: ", str(F1))
    # my_tree.print_tree()
    # for row in testing_data:
    #     print ("Actual: %s. Predicted: %s" %
    #            (row[-1], my_tree.print_leaf(my_tree.classify(row, my_tree.tree))))

# Next steps
# - add support for missing (or unseen) attributes
# - prune the tree to prevent overfitting