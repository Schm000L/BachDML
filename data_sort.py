def makeSet(file):
    l1=[]
    for line in open(file, 'r').readlines():
        set = []
        line = line[2:-1]+',' +line[:1]
        # print(line)
        line = line.split(',')
        for word in line:
            if isfloat(word):
                set.append(float(word))
            else:
                if word is '':
                    continue
                elif word is ' ':
                    continue
                set.append(word)
        if len(set) == 9:
            l1.append(set)
        else: continue
    return l1

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