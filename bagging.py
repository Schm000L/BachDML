import data_sort
from dt_thread import TreeThread
from random import randint
import time
import math
import dt

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

def hard_extraction(dataSet, number_of_workers, worker_number, overlap):
    data = []
    start_position = math.floor(((worker_number - 1) / number_of_workers) * len(dataSet))
    end_position = math.ceil(start_position + len(dataSet)/number_of_workers + overlap)
    
    if end_position > len(dataSet):
        new_end = end_position - len(dataSet)
        for i in range(start_position, len(dataSet)):
            data.append(dataSet[i])
        for i in range(0, new_end):
            data.append(dataSet[i])
    else:
        for i in range(start_position, end_position):
            data.append(dataSet[i])
    
    return data

def test(num_work, threads):
    accuracy = 0
    correct_prediction = 0
    TPover50 = 0
    FPover50 = 0
    FNover50 = 0
    TPunder50 = 0
    FNunder50 = 0
    FPunder50 = 0

        # Test on testing_data
    for i in range(0, len(test_data)):
        classed = {}
        prediction = []
        for j in range(0, len(threads)):
            prediction.append(threads[j].query(test_data[i]))

        while len(prediction) < len(threads):
            time.sleep(1)

        for predicted in prediction:
            label = max(predicted, key=lambda key: predicted[key])
            if label not in classed:
                classed[label] = 0
            classed[label] += 1
        correct_lable = str(test_data[i][number_of_features - 1])
        print(correct_lable)
        predicted_lable = list(classed.keys())[list(classed.values()).index(max(classed.values()))]
        print(predicted_lable, max(classed.values()))

        if predicted_lable in correct_lable:
            correct_prediction += 1
            if ">50K" in predicted_lable: # "acc"
                TPover50 += 1
            else:
                TPunder50 += 1
        else:
            if ">50K" in predicted_lable: # "acc"
                FPover50 += 1
                FNunder50 += 1
            else:
                FNover50 += 1
                FPunder50 += 1

    precision = TPover50 / (TPover50 + FPover50)
    recall = TPover50 / (TPover50 + FNover50)
    F1 = 2 * ((precision*recall) /  (precision + recall))
    accuracy = correct_prediction / len(test_data)

    file = open("HDP2BIG" + "workers" + str(num_work) + ".txt", "w+")
    print("Accuracy: ", str(accuracy))
    print("Precision: ", str(precision))
    print("Recall: ", str(recall))
    print("F1 score: ", str(F1))
    print("Execution time (s): ", time.time() - start_time)
    file.write("Number of workers: "+ str(num_work)+ "\n")
    file.write("Accuracy: "+ str(accuracy) + "\n")
    file.write("Execution time (s): " + str(time.time() - start_time) + "\n")
    file.write("Precision: "+ str(precision)+ "\n")
    file.write("Recall: "+ str(recall)+ "\n")
    file.write("F1 score: "+ str(F1)+ "\n")
    file.close()
    print(str(i))

def main_loop():
    a = 0
    number_of_workers_list = [1,2,5, 10, 20, 50, 100, 200, 400]
    while a < len(number_of_workers_list):
        # Create threads
        if not hdp:
            print("Running bagging!")
            number_of_workers = number_of_workers_list[-1]

            threads = []
            for i in range(0, number_of_workers):
                threads.append(TreeThread(str(i), extract_training_data(training_data)))
                if i+1 == number_of_workers_list[a]:
                    test(i+1, threads)
                    a+=1
        else:
            print("Running HDP!")
            number_of_workers = number_of_workers_list[a]
            threads = []
            for i in range(0, number_of_workers):
                threads.append(TreeThread(str(i), hard_extraction(training_data, number_of_workers, i, 10)))
            test(number_of_workers_list[a],threads)
            start_time = time.time()
            a+=1
        
hdp = True
start_time = time.time()
number_of_features = 15
training_data = data_sort.makeSet("datasets/adult_data_big.txt", number_of_features)
test_data = data_sort.makeSet("datasets/adult_data_test.txt", number_of_features)
main_loop()


# Single strong learner
# print("Compare to a single strong learning: ")
# strong_learner = dt.Decision_Tree(training_data)
# print("Running test")
# successful_prediction = 0
# for i in range(0, len(test_data)):
#     correct_lable = str(test_data[i][number_of_features-1])
#     # print("Corr:",correct_lable)
#     # print("Predicted:", list(strong_learner.predict(strong_learner.tree, test_data[i]).keys())[0])
#     if list(strong_learner.predict(strong_learner.tree, test_data[i]).keys())[0] == correct_lable:
#         successful_prediction += 1
# acc = successful_prediction/(len(test_data))
# print("Accuracy of strong learner", str(acc))
    