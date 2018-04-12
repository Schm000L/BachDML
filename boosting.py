import data_sort
from dt_thread import TreeThread
# from dt import Decision_Tree
from random import randint
import time

startTime = time.time()
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

number_of_workers = 10
data_for_workers = []
number_of_features = 15
training_data = data_sort.makeSet("adult_data.txt", number_of_features)
weights = []

for i in range(0, len(training_data)):
    weights.append(1)

# TODO: Ta hänsyn till weights
def extract_training_data(dataSet):
    data = []
    for n in range(0, int(round(0.6*len(dataSet)))):
        data.append(dataSet[randint(0, len(dataSet)-1)])
    return data

# Create threads
threads = []
for i in range(0, number_of_workers):
    # 1. Train thread
    # 2. Evaluate and weight training_data

    threads.append(TreeThread(str(i), extract_training_data(training_data)))
    # threads.append(TreeThread(str(i), "abalone_train.txt", number_of_features))

# # Start threads
# for i in range(0, len(threads)):
#     # print(threads[i].threadID + " started")
#     threads[i].start()

test_data_abalone = data_sort.makeSet("adult_data_test.txt", number_of_features)

accuracy = 0
correct_prediction = 0

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
    correct_lable = str(test_data_abalone[i][number_of_features-1])
    print(correct_lable)
    # print(len(classed.values()))
    # print(max(classed.values()))
    predicted_lable = list(classed.keys())[list(classed.values()).index(max(classed.values()))]
    print(predicted_lable, max(classed.values()))
    if predicted_lable in correct_lable:
        correct_prediction += 1
accuracy = correct_prediction/len(test_data_abalone)
print("accuracy", str(accuracy))

print("Execution time (s): ", time.time() - startTime)

# Single strong learner
# print("Compare to a single strong learning: ")
# strong_learner = Decision_Tree(training_data)

# successful_prediction = 0
# for i in range(0, len(test_data_abalone)):
#     correct_lable = str(test_data_abalone[i][number_of_features-1])
#     # print("Corr:",correct_lable)
#     # print("Predicted:", list(strong_learner.predict(strong_learner.tree, test_data_abalone[i]).keys())[0])
#     if list(strong_learner.predict(strong_learner.tree, test_data_abalone[i]).keys())[0] == correct_lable:
#         successful_prediction += 1
# acc = successful_prediction/(len(test_data_abalone))
# print("Accuracy of strong learner", str(acc))
    