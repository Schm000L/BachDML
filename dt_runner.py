import decision_tree
import threading
import time
import data_sort

exitFlag = 0

class newWorker(threading.Thread):

    def __init__(self, threadID, name, counter):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter
    def run(self):
        print("Starting "+ self.name)
        print_time(self.name, 5 , self.counter)
        print("Existing " + self.name)

def print_time(threadName, counter, delay):
    while counter:
        if exitFlag:
            threadName.exit()
        time.sleep(delay)
        print("%s: %s" % (threadName, time.ctime(time.time())))
        counter -= 1

# Create new threads
thread1 = newWorker(1, "Thread-1", 1)
thread2 = newWorker(2, "Thread-2", 2)

# Start new Threads
thread1.start()
thread2.start()


print("Exiting Main Thread")

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



# Evaluation data
testing_data = [
    ['Green', 3, 'Apple'],
    ['Yellow', 4, 'Apple'],
    ['Red', 2, 'Grape'],
    ['Red', 1, 'Grape'],
    ['Yellow', 3, 'Lemon'],
]


number_of_features = 10
# print(makeSet('scale_train.txt'))
tree_1 = decision_tree.build_tree(data_sort.makeSet('abalone.txt',number_of_features))
# tree_1 = decision_tree.build_tree(makeSet('scale_train.txt'))
# tree_1 = decision_tree.build_tree(training_data_1)
# tree_2 = decision_tree.build_tree(training_data_2)
print("tree_1: ")
# decision_tree.print_tree(tree_1)
decision_tree.test(tree_1, data_sort.makeSet('abalone_text.txt', number_of_features))
# decision_tree.test(tree_1, makeSet('scale_test.txt'))
# decision_tree.test(tree_1, testing_data)
print("")

# print("Testing tree_2:")
# decision_tree.print_tree(tree_2)
# decision_tree.test(tree_2, testing_data)
