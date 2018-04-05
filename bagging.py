from dt_thread import TreeThread
from random import randint

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

threads = []
thread1 = TreeThread("1", extract_training_data(training_data))
thread2 = TreeThread("2", extract_training_data(training_data))

threads.append(thread1)
threads.append(thread2)

# Start threads
for i in range(0, len(threads)):
    print(threads[i].threadID + " started")
    threads[i].start()

for i in range(0, len(test_data)):
    classed = {}   
    for j in range(0, len(threads)):
        # temp = []
        # temp.append(test_data[i])
        # threads[j].query(temp)
        predicted = threads[j].query(test_data[i])
        label = max(predicted, key=lambda key: predicted[key])
        if label not in classed:
            classed[label] = 0
        classed[label] += 1
    print(classed)
    
#thread1.start()
#thread2.start()

#thread1.query(test_data)
#thread2.query(test_data)