import data_sort
from dt_thread import TreeThread
# from dt import Decision_Tree
from random import randint
import time
import math

start_time = time.time()
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

# TODO: Ta hänsyn till weights
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

def main_loop():

     a = 0
     number_of_workers_list = [1,2,5, 10, 20, 50, 100, 200, 400]
     while a < len(number_of_workers_list):
        # Create threads
        number_of_workers = number_of_workers_list[a]

        threads = []
        for i in range(0, number_of_workers):
            # threads.append(TreeThread(str(i), extract_training_data(training_data)))
            threads.append(TreeThread(str(i), hard_extraction(training_data, number_of_workers, i, 10)))
            # threads.append(TreeThread(str(i), "abalone_train.txt", number_of_features))

        # # Start threads
        # for i in range(0, len(threads)):
        #     # print(threads[i].threadID + " started")
        #     threads[i].start()

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
            # print(len(classed.values()))
            # print(max(classed.values()))
            predicted_lable = list(classed.keys())[list(classed.values()).index(max(classed.values()))]
            print(predicted_lable, max(classed.values()))

            if predicted_lable in correct_lable:
                correct_prediction += 1
                if ">50K" in predicted_lable:
                    TPover50 += 1
                else:
                    TPunder50 += 1
            else:
                if ">50K" in predicted_lable:
                    FPover50 += 1
                    FNunder50 += 1
                else:
                    FNover50 += 1
                    FPunder50 += 1

        precission_over_50 = TPover50 / (TPover50 + FPover50)
        recall_over_50 = TPover50 / (TPover50 + FNover50)
        precission_under_50 = TPunder50 / (TPunder50 + FPunder50)
        recall_under_50 = TPunder50 / (TPunder50 + FNunder50)
        F1 = 2 * ((precission_over_50 + precission_under_50) / 2 * (recall_over_50 + recall_under_50) / 2) / (
                    (precission_over_50 + precission_under_50) / 2 + (recall_over_50 + recall_under_50) / 2)
        accuracy = correct_prediction / len(test_data)

        file = open("workers" + str(number_of_workers) + ".txt", "w+")
        print("Accuracy: ", str(accuracy))
        print("Precision: ", str(precission_over_50))
        print("Recall: ", str(recall_over_50))
        print("F1 score: ", str(F1))
        print("Execution time (s): ", time.time() - start_time)
        file.write("Number of workers: "+ str(number_of_workers)+ "\n")
        file.write("Accuracy: "+ str(accuracy) + "\n")
        file.write("Execution time (s): " + str(time.time() - start_time) + "\n")
        file.write("Precision: "+ str(precission_over_50)+ "\n")
        file.write("Recall: "+ str(recall_over_50)+ "\n")
        file.write("F1 score: "+ str(F1)+ "\n")
        file.close()
        print(str(i))
        a+=1

number_of_features = 15
training_data = data_sort.makeSet("adult_data.txt", number_of_features)
test_data = data_sort.makeSet("adult_data_test.txt", number_of_features)
main_loop()


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
    