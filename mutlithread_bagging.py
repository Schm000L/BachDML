import _thread
import dt



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
local_tree = dt(training_data_1)
local_tree.print_tree()
tree1 = _thread.start_new_thread(decision_tree.build_tree, (training_data_1, ))
tree2 = _thread.start_new_thread(decision_tree.build_tree, (training_data_2, ))


