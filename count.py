import data_sort

number_of_columns = 7
training_data = data_sort.makeSet("datasets/car_split_train.txt", number_of_columns)
test_data = data_sort.makeSet("datasets/car_split_test.txt", number_of_columns)
label_positive = "acc"
label_negative = "unacc"

print("Training data")
over_tr = 0
under_tr = 0
for row in training_data:
    if row[-1] == label_positive:
        over_tr += 1
    elif row[-1] == label_negative:
        under_tr += 1
print(label_positive, over_tr)
print(label_negative, under_tr)
print("Sum", over_tr+under_tr)

print("Testing data")
over = 0
under = 0

for row in test_data:
    if row[-1] == label_positive:
        over += 1
    elif row[-1] == label_negative:
        under += 1
print(label_positive, over)
print(label_negative, under)
print("Sum", over+under)

print("Ratio:", (over_tr+over)/(under_tr+under+over_tr+over))