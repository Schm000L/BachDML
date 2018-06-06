import math
from random import randint

file = "datasets/car_full.txt"
file_train = open("car_split_train" + ".txt", "w+")
file_test = open("car_split_test" + ".txt", "w+")

test_lines = 347
train_lines = 1381
lines = []
for line in open(file, 'r').readlines():
    lines.append(line)

while test_lines > 0:
    file_test.write(lines.pop(randint(0, test_lines+train_lines-1)))
    test_lines += -1

for line in lines:
    file_train.write(line)


