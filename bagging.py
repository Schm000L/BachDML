import data_sort
from dt_thread import TreeThread
from random import randint
import time

training_data = [
    ['Green', 3, 'Apple'],
    ['Red', 3, 'Apple'],
    ['Red', 1, 'Grape'],
    ['Red', 2, 'Grape'],
    ['Yellow', 3, 'Lemon'],
    ['Yellow', 3, 'Apple'],
    ['Red', 1, 'Grape'],
    ['Red', 2, 'Grape'],
    ['Green', 1, 'Grape'],
    ['Yellow', 2, 'Lemon']
]

test_data = [
    ['Green', 1, 'Grape'],
    ['Yellow', 2, 'Lemon'],
    ['Green', 2, 'Apple'],
    ['Green', 4, 'Apple']
]


def extract_training_data(dataSet):
    data = []
    for n in range(0, int(round(0.6*len(dataSet)))):
        data.append(dataSet[randint(0, len(dataSet)-1)])
    return data

# Create threads
threads = []
for i in range(0, 11):
    threads.append(TreeThread(str(i), extract_training_data(data_sort.makeSet("adult_data.txt", 15))))

# Start threads
for i in range(0, len(threads)):
    print(threads[i].threadID + " started")
    threads[i].start()

test_data_abalone = data_sort.makeSet("adult_data_test.txt", 15)

# Test on testing_data
for i in range(0, len(test_data_abalone)):
    classed = {}
    prediction = []   
    for j in range(0, len(threads)):
        prediction.append(threads[j].query(test_data_abalone[i]))
    
    while len(prediction) < len(threads):
        sleep(1) 
    
    for predicted in prediction:
        label = max(predicted, key=lambda key: predicted[key])
        if label not in classed:
            classed[label] = 0
        classed[label] += 1
    print(str(test_data_abalone[i]))
    print(classed)
