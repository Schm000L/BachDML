import decision_tree

def test(dt, rows):
    for row in rows:
        print ("Actual: %s. Predicted: %s" % (row[-1], decision_tree.print_leaf(decision_tree.classify(row, dt))))

training_data_1 = [
    ['Green', 3, 'grapf', 'Apple'],
    ['Yellow', 3, 'apf', 'Apple'],
    ['Red', 1, 'grapf', 'Grape'],
    ['Red', 1, 'grapf','Grape'],
    ['Yellow', 3, 'lepf', 'Lemon']
]

training_data_2 =  [
    ['Yellow', 3, 'grapf', 'Apple'],
    ['Red', 1, 'grapf', 'Grape'],
    ['Red', 1, 'grapf','Grape'],
    ['Yellow', 3, 'grapf', 'Lemon']
]

tree_1 = decision_tree.build_tree(training_data_1)
tree_2 = decision_tree.build_tree(training_data_2)

# Evaluation data
testing_data = [
    ['Green', 3, 'apf', 'Apple'],
    ['Yellow', 4, 'grapf', 'Apple'],
    ['Red', 2, 'grapf', 'Grape'],
    ['Red', 1, 'apf','Grape'],
    ['Yellow', 3, 'lepf', 'Lemon'],
]

print("tree_1: ")
decision_tree.print_tree(tree_1)
test(tree_1, testing_data)
print("")

print("Testing tree_2:")
decision_tree.print_tree(tree_2)
test(tree_2, testing_data)