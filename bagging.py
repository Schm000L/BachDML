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


def extract_training_data(training_data):
    data = []
    for n in range(0, 6):
        data.append(training_data[randint(0, len(training_data)-1)])
    return data

# Create threads
threads = []
for i in range(0, 5):
    threads.append(TreeThread(str(i), extract_training_data(training_data)))

# Start threads
for i in range(0, len(threads)):
    print(threads[i].threadID + " started")
    threads[i].start()

# Test on testing_data
for i in range(0, len(test_data)):
    classed = {}
    prediction = []   
    for j in range(0, len(threads)):
        prediction.append(threads[j].query(test_data[i]))
    
    while len(prediction) < len(threads):
        sleep(1) 
    
    for predicted in prediction:
        label = max(predicted, key=lambda key: predicted[key])
        if label not in classed:
            classed[label] = 0
        classed[label] += 1
    print(classed)