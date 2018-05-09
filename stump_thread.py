#import _thread
import threading
from decision_stump import Decision_Stump
import data_sort
from random import randint

#Creating new worker thread
class StumpThread(threading.Thread):

    def __init__(self, threadID,  dataSet):
        threading.Thread.__init__(self)
        self.start()
        self.threadID = threadID
        self.decision_stump = Decision_Stump(dataSet)

    def query(self, query_data):
        # print(self.threadID + " queried")
        return self.decision_stump.predict(self.decision_stump.stump, query_data)

    def binary_query(self, query_data):
        # print(self.threadID + " queried")
        return self.decision_stump.binary_prediction(self.decision_stump.stump, query_data)

    def print_stump(self):
        self.decision_stump.print_stump()


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

    local_stump = Decision_Stump(training_data_1)
    local_stump.print_stump()
    
    thread1 = StumpThread("1", training_data_1)
    thread2 = StumpThread("2", training_data_2)

    thread1.query(training_data_2)
    thread2.query(training_data_1)


