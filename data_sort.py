def isfloat(value):
    try:
        float(value)
        return True
    except:
        return False

def isint(value):
  try:
    int(value)
    return True
  except:
    return False

def makeSet(file, length):
    l1=[]
    for line in open(file, 'r').readlines():
        set = []
        # print(line)
        line = line.split(',')
        for word in line:
            word = word.replace('[', '')
            word = word.replace(']', '')
            word = word.replace('"', '')
            word = word.replace('\n', '')
            word = word.replace(" ", "")
            word = word.replace("'", "")
            word = word.replace(".", "")

            if isfloat(word):
                set.append(float(word))
            else:
                if word is '':
                    continue
                elif word is ' ':
                    continue
                set.append(word)
        if len(set) == length:
            l1.append(set)
        else: continue
    return l1

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

def split(data_set):
    X = []
    y = []
    rows = data_set.copy()
    for row in rows:
        y.append(row[-1])
        row.remove(row[-1])
        # row[-1] = 15
        X.append(row)
    return X, y
        


# print(makeSet("abalone_train.txt", 9))