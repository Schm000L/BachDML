import data_sort

training_data = data_sort.makeSet("adult_data_big.txt", 15)

label1 = 0
label2 = 0

file = open("adult_data_unbiased.txt", "w+")
for a in training_data:
    if label1 != 0:
        if a[14] != label1[14]:
            label2 = a
            file.write(str(label1) + "\n")
            file.write(str(label2) + "\n")
            label1 = 0
            label2 = 0
    else: label1 = a

file.close()