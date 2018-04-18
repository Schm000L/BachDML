import data_sort
from dt_thread import TreeThread
from random import randint
import math
import time

# FEL: Varv 1 funkar till synes felfritt
# Varv 2 ger ERROR 3 ggr
# Varv 2 kraschar sedan vid calculate_weights

start_time = time.time()
number_of_workers = 2
data_for_workers = []

number_of_features = 15

training_data = data_sort.makeSet("adult_data_big.txt", number_of_features)
training_data, labels = data_sort.binaryfy(training_data)

alpha = []
epsilon = 0.001
start_weight = 1/len(training_data)
weights = []
# print(start_weight)
for n in range(0, len(training_data)):
    weights.append(start_weight)

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


def calculate_weights(weights, error_rate, alpha, data_set,  predictions):
    # TODO: Properly calculate Z, maybe add predicted[] instead of querying
    temp_weights = weights.copy()
    ret = []
    # Z = error_rate*math.exp(alpha) + (1-error_rate)*math.exp(-alpha)
    Z = 2*math.sqrt((error_rate+epsilon)*(1-(error_rate+epsilon)))
    for i in range(0, len(temp_weights)):
        # print("Old:", temp_weights[i])    
        ret.append((temp_weights[i] * math.exp(-1*data_set[i][-1] * alpha * predictions[i])) / Z)
        # print("Checking preds")
        # print((math.exp(-1*alpha * data_set[i][-1] * predictions[i])))
        
        # if temp_weights[i] < 0:
        #     print('Weighing error', temp_weights[i])
        #     print("Alpha", alpha, "Prediction", predictions[i], "Corr", data_set[i][-1])
        #     print("Exp", math.exp(data_set[i][-1] * alpha * predictions[i]))
    # print(ret)
    return ret

# TODO: Ta hänsyn till weights
def extract_training_data(data_set):
    data = []
    for n in range(0, int(round(0.6*len(data_set)))):
        data.append(data_set[randint(0, len(data_set)-1)])
    return data

def calculate_error_rate(worker_number, weights, data_set, predictions):
    error_rate = 0
    num_errors = 0
    for i in range(0, len(data_set)):
        if data_set[i][-1] != predictions[i]:
            error_rate += weights[i]
            num_errors += 1
            # print("Error detected:", "predicted", predictions[i], "was", data_set[i][-1])
            # print("Weight:", weights[i])
    if error_rate == 0 and num_errors != 0:
        print("Rounding error")
    # if num_errors == 0:
    #     print("Woah, we got all of them right!!!")
    return error_rate

def calculate_alpha(worker_number, error_rate):
    # print("Error rate", error_rate)
    return 1/2*math.log((1-error_rate+epsilon)/(error_rate+epsilon))

# indxs = []
def extract_weighted_data(data_set, weights):
    # indx = []
    tmp_weights = weights.copy() # For scoping
    data = []
    for i in range(0, math.floor(0.6*len(data_set))):
        # index = tmp_weights.index(max(tmp_weights))
        # index, max_value = max(enumerate(tmp_weights), key=operator.itemgetter(1))
        index = tmp_weights.index(max(tmp_weights))
        # indx.append(index)
        data.append(data_set[index])
        tmp_weights[index] = 0
    # indxs.append(indx)
    return data

def make_predictions(worker_number, data_set):
    predictions = []
    for n in range(0, len(data_set)):
        predictions.append(threads[worker_number].binary_query(data_set[n]))
    return predictions

# preds  = []
# dats = []
# wghts = []

# Create threads, train and evaluate them
threads = []
for i in range(0, number_of_workers):
    # wghts.append(weights)
    print("Training", i+1, "at time", time.time()-start_time)
    w_sum = 0
    # for w in weights:
    #     w_sum += w
    # print("Sum of all weights", w_sum)
    threads.append(TreeThread(str(i), extract_weighted_data(training_data, weights)))
    # dats.append(extract_weighted_data(training_data, weights))
    print("Getting predictions")
    predictions = make_predictions(i, training_data)    
    # preds.append(predictions)
    print("Calculating error_rate")
    error = calculate_error_rate(i, weights, training_data, predictions)
    # print("error:", i, error)
    print("Calculating alpha")
    alpha.append(calculate_alpha(i, error))
    # print('Alpha', alpha[i])
    if i != number_of_workers-1:
        print("Calculating new weights")
        weights = calculate_weights(weights, error, alpha[i], training_data, predictions)
    # print("Alpha:", alpha[i])
    # threads.append(TreeThread(str(i), "abalone_train.txt", number_of_features)

# diff = 0
# for m in range(0, number_of_workers):
#     for n in range(0, number_of_workers):
#         if dats[m] != dats[n]:
#             diff += 1
# print("diff", diff)

# # Start threads
# for i in range(0, len(threads)):
#     # print(threads[i].threadID + " started")

# def run_test(test_data, thread_number):

test_data = data_sort.makeSet("adult_data_test.txt", number_of_features)
test_data, labels = data_sort.binaryfy(test_data)

# correct_prediction = 0

# false_negative_x == false_positive_y
accuracy = 0
correct_prediction = 0
true_positive_minus = 0
false_positive_minus = 0
false_negative_minus = 0
true_positive_plus = 0
false_negative_plus = 0
false_positive_plus = 0


print("RUNNING TEST")
# Test on test_data
for i in range(0, len(test_data)):
    prediction  = 0 
    for j in range(0, len(threads)):
        prediction += alpha[j] * threads[j].binary_query(test_data[i])
    
    # Make sure every thread has answered
    # time.sleep(0.00005)

    # # Wait to ensure that every worker has answered
    # while len(prediction) < len(threads):
    #     time.sleep(1) 
    
    # The sign of prediction decides the models final prediction
    if prediction > 0:
        prediction = 1
    elif prediction <= 0:
        prediction = -1

    if prediction == test_data[i][-1]:
        correct_prediction += 1
    
        if prediction == 1:
            true_positive_plus += 1
        else:
            true_positive_minus += 1
    
    else:
        # predicted 1, correct is -1
        if prediction == 1:
            false_positive_plus += 1
            false_negative_minus += 1

        # predicted -1, correct is 1
        else:
            false_positive_minus += 1
            false_negative_plus += 1

    # else:
    #     print("Oops predicted wrong")
    #     if prediction == -1:
    #         print("Predicted: ", labels[0])
    #         print("on", test_data[i])
    #     else:
    #         print("Predicted: ", labels[1])
    #         print("on", test_data[i])
precission_plus = true_positive_plus / (true_positive_plus + false_positive_plus)
recall_plus = true_positive_plus / (true_positive_plus + false_negative_plus)
precission_minus = true_positive_minus/(true_positive_minus + false_positive_minus)
recall_minus = true_positive_minus/(true_positive_minus+false_negative_minus)
F1 = 2*((precission_plus+precission_minus)/2 * (recall_plus+recall_minus)/2) / ((precission_plus+precission_minus)/2 + (recall_plus+recall_minus)/2)
accuracy = correct_prediction/len(test_data)

print("Accuracy: ", str(accuracy))
print("Precision: ", str(precission_plus))
print("Recall: ", str(recall_minus))
print("F1 score: ", str(F1))
print("Execution time (s): ", time.time() - start_time)

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


# Make training data on the form: feature feature ... lable (-1, 1)
# Label contains the actual labels
