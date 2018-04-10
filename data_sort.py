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

# print(makeSet("abalone_train.txt", 9))