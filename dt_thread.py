#import _thread
import threading
from dt import Decision_Tree
import data_sort
from random import randint

#Creating new worker thread
class TreeThread(threading.Thread):

    def __init__(self, threadID,  dataSet):
    # def __init__(self, threadID,  stringToFile, nbrOfFeatures):
        # print("init")
        threading.Thread.__init__(self)
        self.start()
        self.threadID = threadID
        self.decision_tree = Decision_Tree(dataSet)
        # data = self.extract_training_data(data_sort.makeSet(stringToFile, nbrOfFeatures))
        # self.decision_tree = Decision_Tree(data)
    # def __init__def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, verbose=None):
    #     threading.Thread.__init__(self, group=group, target=target, name=name,verbose=verbose)
    #     self.args = args
    #     self.kwargs = kwargs
    #     return

    def query(self, query_data):
        # print(self.threadID + " queried")
        return self.decision_tree.predict(self.decision_tree.tree, query_data)

    def print_tree(self):
        self.decision_tree.print_tree()

    def extract_training_data(self, dataSet):
        data = []
        for n in range(0, int(round(0.6 * len(dataSet)))):
            data.append(dataSet[randint(0, len(dataSet) - 1)])
        return data


if __name__ == "__main__":
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

    local_tree = Decision_Tree(training_data_1)
    local_tree.print_tree()
    
    thread1 = TreeThread("1", training_data_1)
    thread2 = TreeThread("2", training_data_2)

    thread1.start()
    thread2.start()

    thread1.query(training_data_2)
    thread2.query(training_data_1)

    #tree1 = _thread.start_new_thread(decision_tree.build_tree, (training_data_1, ))
    #tree2 = _thread.start_new_thread(decision_tree.build_tree, (training_data_2, ))


