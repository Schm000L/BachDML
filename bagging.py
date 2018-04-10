import data_sort
from dt_thread import TreeThread
from random import randint
from datetime import datetime
import time

startTime = datetime.now()
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

numOfWorkers = 20
dataForWorkers = []
nbrOfFeatures = 15

def extract_training_data(dataSet):
    data = []
    for n in range(0, int(round(0.6*len(dataSet)))):
        data.append(dataSet[randint(0, len(dataSet)-1)])
    return data

# Create threads
threads = []
for i in range(0, numOfWorkers):
    threads.append(TreeThread(str(i), extract_training_data(data_sort.makeSet("adult_data.txt", nbrOfFeatures))))
    # threads.append(TreeThread(str(i), "abalone_train.txt", nbrOfFeatures))

# # Start threads
# for i in range(0, len(threads)):
#     # print(threads[i].threadID + " started")
#     threads[i].start()

test_data_abalone = data_sort.makeSet("adult_data_test.txt", nbrOfFeatures)

accuracy = 0
correctVSAll = [0,0]

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
    correctLable = str(test_data_abalone[i][nbrOfFeatures-1])
    print(correctLable)
    # print(len(classed.values()))
    # print(max(classed.values()))
    predictedLable = list(classed.keys())[list(classed.values()).index(max(classed.values()))]
    print(predictedLable, max(classed.values()))
    if predictedLable in correctLable:
        correctVSAll[0] += 1
        correctVSAll[1] += 1
    else: correctVSAll[1] += 1
accuracy = correctVSAll[0]/correctVSAll[1]
print("accuracy", str(accuracy))

print(datetime.now() - startTime)