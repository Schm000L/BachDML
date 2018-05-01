import data_sort

number_of_features = 15
training_data = data_sort.makeSet("adult_data_big.txt", number_of_features)
test_data = data_sort.makeSet("adult_data_test.txt", number_of_features)

print("Training data")
over = 0
under = 0
for row in training_data:
    if row[-1] == ">50K":
        over += 1
    elif row[-1] == "<=50K":
        under += 1
print("Over 50k", over)
print("Under 50k", under)
print("Sum", over+under)


print("Testing data")
over = 0
under = 0

for row in test_data:
    if row[-1] == ">50K":
        over += 1
    elif row[-1] == "<=50K":
        under += 1
print("Over 50k", over)
print("Under 50k", under)
print("Sum", over+under)