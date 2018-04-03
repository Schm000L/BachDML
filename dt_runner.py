import decision_tree

training_data_1 = [
    ['Green', 3, 'Apple'],
    ['Red', 3, 'Apple'],
    ['Red', 1, 'Grape'],
    ['Red', 1,'Grape'],
    ['Yellow', 3, 'Lemon']
]

training_data_2 =  [
    ['Yellow', 3, 'Apple'],
    ['Red', 1, 'Grape'],
    ['Red', 1, 'Grape'],
    ['Yellow', 3,'Lemon']
]

tree_1 = decision_tree.build_tree(training_data_1)
tree_2 = decision_tree.build_tree(training_data_2)

# Evaluation data
testing_data = [
    ['Green', 3, 'Apple'],
    ['Yellow', 4, 'Apple'],
    ['Red', 2, 'Grape'],
    ['Red', 1, 'Grape'],
    ['Yellow', 3, 'Lemon'],
]

print("tree_1: ")
decision_tree.print_tree(tree_1)
decision_tree.test(tree_1, testing_data)
print("")

print("Testing tree_2:")
decision_tree.print_tree(tree_2)
decision_tree.test(tree_2, testing_data)