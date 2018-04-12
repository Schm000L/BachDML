import data_sort
from dt_thread import TreeThread
# from dt import Decision_Tree
from random import randint
import math
import time
import numpy
import operator

# startTime = time.time()
# training_data = [
#     ['Green', 3, 'Apple'],
#     ['Red', 3, 'Apple'],
#     ['Red', 1, 'Grape'],
#     ['Red', 2, 'Grape'],
#     ['Yellow', 3, 'Apple'],
#     ['Red', 1, 'Grape'],
#     ['Red', 2, 'Grape'],
#     ['Green', 1, 'Grape']
# ]

# test_data = [
#     ['Green', 1, 'Grape'],
#     ['Green', 2, 'Apple'],
#     ['Green', 4, 'Apple']
# ]

# Kolla https://github.com/jaimeps/adaboost-implementation
def binaryfy(rows):
    label_count = 0
    labels = []
    for row in rows:
        label = row[-1]
        if label in labels:
            if label == labels[0]:
                row[-1] = -1
            elif label == labels[1]:
                row[-1] = 1
        elif len(labels) == 0:
            labels.append(label)
            row[-1] = -1
        elif len(labels) == 1:
            labels.append(label)
            row[-1] = 1
        else:
            print("Something went wrong")
            print("Labels:", labels)
            print(row[-1])
    return rows, labels

number_of_workers = 2
data_for_workers = []

number_of_features = 15

# Make training data on the form: feature feature ... lable (-1, 1)
# Label contains the actual labels
training_data = data_sort.makeSet("adult_data.txt", number_of_features)
training_data, labels = binaryfy(training_data)

alpha = []
start_weight = 1/len(training_data)
weights = []
print(start_weight)
for n in range(0, len(training_data)):
    weights.append(start_weight)

# print("START")
# print(weights)
# def loss_function(test_data):
#     loss = 0
#     for n in range(0, len(test_data)):
#         m = 0
#         for k in range(0, len(threads)):        
#             m += alpha[k]*threads[k].binary_query(test_data[n])
#         m = m*test_data[n][-1]
#         loss += exp(-m)
#     return loss

# def calculate_weight(old_weight, row, worker_number):
#     Z = 1
#     weight = old_weight * math.exp(row[-1]* alpha[worker_number]*threads[worker_number].binary_query(data_set[i])) / Z
#     return weight

def weighing(weights, error_rate, alpha, data_set,  predictions):
    # TODO: Properly calculate Z, maybe add predicted[] instead of querying
    temp_weights = weights.copy()
    Z = error_rate*math.exp(alpha) + (1-error_rate)*math.exp(-alpha)
    for i in range(0, len(temp_weights)):
        # print("Old:", temp_weights[i])    
        temp_weights[i] = temp_weights[i] * math.exp(data_set[i][-1] * alpha * predictions[i]) / Z
        # if temp_weights[i] < 0:
        #     print('Weighing error', temp_weights[i])
        #     print("Alpha", alpha, "Prediction", predictions[i], "Corr", data_set[i][-1])
        #     print("Exp", math.exp(data_set[i][-1] * alpha * predictions[i]))

    return temp_weights

# TODO: Ta hänsyn till weights
def extract_training_data(data_set):
    data = []
    for n in range(0, int(round(0.6*len(data_set)))):
        data.append(data_set[randint(0, len(data_set)-1)])
    return data

def calculate_error_rate(worker_number, weights, data_set, predictions):
    error_rate = 0
    for i in range(0, len(data_set)):
        if data_set[i][-1] != predictions[i]:
            error_rate += weights[i]
            # print("Error detected:", "predicted", predictions[i], "was", data_set[i][-1])
            # print("Weight:", weights[i])
    return error_rate

def calculate_alpha(worker_number, error_rate):
    print("Error rate", error_rate)
    return 1/2*math.log((1-error_rate)/error_rate)


def extract_weighted_data(data_set, weights):
    tmp_weights = weights.copy() # For scoping
    data = []
    for i in range(0, math.floor(0.6*len(data_set))):
        # index = tmp_weights.index(max(tmp_weights))
        # index, max_value = max(enumerate(tmp_weights), key=operator.itemgetter(1))
        index = tmp_weights.index(max(tmp_weights))
        data.append(data_set[index])
        tmp_weights[index] = 0
    return data

def make_predictions(worker_number, data_set):
    predictions = []
    for n in range(0, len(data_set)):
        predictions.append(threads[worker_number].binary_query(data_set[n]))
    return predictions

# Create threads
threads = []
for i in range(0, number_of_workers):
    # 1. Train thread
    # 2. Evaluate and weight training_data
    threads.append(TreeThread(str(i), extract_weighted_data(training_data, weights)))
    predictions = make_predictions(i, training_data)    

    error = calculate_error_rate(i, weights, training_data, predictions)
    print("error:", error)
    alpha.append(calculate_alpha(i, error))
    weights = weighing(weights, error, alpha[i], training_data, predictions)
    print("Alpha:", alpha[i])
    # threads.append(TreeThread(str(i), "abalone_train.txt", number_of_features))

# # Start threads
# for i in range(0, len(threads)):
#     # print(threads[i].threadID + " started")

# def run_test(test_data, thread_number):


# test_data = data_sort.makeSet("adult_data_test.txt", number_of_features)
# test_data, labels = binaryfy(test_data)

# accuracy = 0
# correct_prediction = 0

# # Test on testing_data
# for i in range(0, len(test_data)):
#     classed = {}
#     prediction = []
#     for j in range(0, len(threads)):
#         prediction.append(threads[j].query(test_data[i]))
    
#     while len(prediction) < len(threads):
#         sleep(1) 
    
#     for predicted in prediction:
#         label = max(predicted, key=lambda key: predicted[key])
#         if label not in classed:
#             classed[label] = 0
#         classed[label] += 1
#     correct_lable = test_data[i][number_of_features-1]
#     print("Correct label:", correct_lable)
#     # print(len(classed.values()))
#     # print(max(classed.values()))
#     predicted_lable = list(classed.keys())[list(classed.values()).index(max(classed.values()))]
#     print("Predicted label", predicted_lable)
#     print("-------------------------------------------------")
#     if predicted_lable == correct_lable:
#         correct_prediction += 1
# accuracy = correct_prediction/len(test_data)
# print("accuracy", str(accuracy))

# print("Execution time (s): ", time.time() - startTime)

# Single strong learner
# print("Compare to a single strong learning: ")
# strong_learner = Decision_Tree(training_data)

# successful_prediction = 0
# for i in range(0, len(test_data)):
#     correct_lable = str(test_data[i][number_of_features-1])
#     # print("Corr:",correct_lable)
#     # print("Predicted:", list(strong_learner.predict(strong_learner.tree, test_data[i]).keys())[0])
#     if list(strong_learner.predict(strong_learner.tree, test_data[i]).keys())[0] == correct_lable:
#         successful_prediction += 1
# acc = successful_prediction/(len(test_data))
# print("Accuracy of strong learner", str(acc))
    # Should be possible to make this better

# Calculate error
# for row in training_data:
#     prediction = 0    
#     for j in range(0, len(threads)):
#         prediction += threads[j].binary_query(test_data[i])
#     if prediction > 0:
#         prediction = 1
#     elif prediction <= 0:
#         prediction = -1