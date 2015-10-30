import csv
import time
import math
import operator
import threading
from collections import defaultdict

# Parsing train dataser
train = list (csv.reader(open('data/train.csv', 'rb'), delimiter = ','))
del train[0] #deletion of the string header

# Parsing test dataset
test = list (csv.reader(open('data/test.csv', 'rb'), delimiter = ','))
del test[0]

# initializing resut array
result=[]

# Calculating number of users
nUsers=len(train)

# Creating the UserEvaluatedList: the list of items already evaluated by an user
uel = defaultdict(list)
for line in train:
        line = map (int, line)
        uel[line[0]].append(line[1])

# Parsing the item feature list
byfeature = list(csv.reader(open('data/icm.csv', 'rb'), delimiter = ','))
del byfeature[0]
ifl = {}
for elem in byfeature:
    elem = map (int, elem)
    if not (elem[0]) in ifl:
        ifl[elem[0]]=[]
    ifl[elem[0]].append(elem[1])

# User feature avg rating
ufr={}
ufc={}
urc={}
ufc = defaultdict(lambda: 0.0, ufc)
ufr = defaultdict(lambda: 0.0, ufr)
urc = defaultdict(lambda: 0,urc)
for elem in train:
    elem=map(int, elem)
    u=elem[0]
    i=elem[1]
    r=elem[2]
    urc[u]=urc[u]+1
    if not i in ifl:
        continue
    for f in ifl[i]:
        ufr[(u,f)]=(ufr[(u,f)]+float(r))/2
        ufc[(u,f)]=ufc[(u,f)]+1

## TODO fare la top-N personalizzata con la normalizzazione del voto basata sulle medie valutazioni dell'utente per una certa feature

personalizedTopN={}
dictTopN5=dict(list(csv.reader(open('data/topN.csv', 'rb'), delimiter = ','))[:3100])

for elem in test     
    u=int(elem[0])
    for i in dictTopN5:
        i=int(i)
        personalizedTopN[(u,i)]=math.log(float(dictTopN5[str(i)]))
        if not i in ifl:
            continue
        for f in ifl[i]:
            if not (ufc[(u,f)]==0 or urc[u]==0 or u not in urc or (u,f) not in ufc):
                personalizedTopN[(u,i)]=personalizedTopN[(u,i)]+ufr[(u,f)]/float(float(ufc[(u,f)])/urc[u])

class myThread (threading.Thread):
    def __init__(self, threadID):
        threading.Thread.__init__(self)
        self.threadID = threadID
    def run(self):
        personalizedSplit(self.threadID)

topNPersonalized=sorted(personalizedTopN.items(), key=lambda x:x[1], reverse=True)

for elem in test:
    elem=map(int,elem)
    count=0
    iterator=0
    recommendetions=''
    while count<5:
        if (topNPersonalized[iterator][0][0]==elem[0]):
            if not (topNPersonalized[iterator][0][1] in uel[elem[0]]):
                recommendetions=recommendetions+(str(topNPersonalized[iterator][0][1])+' ')
                count=count+1
        iterator=iterator+1
    elem.append(recommendetions)
    result.append(elem)

## Writing Results
with open ('data/result.csv', 'w') as fp:
     writer = csv.writer(fp, delimiter=',')
     writer.writerow(["userId,testItems"])
     writer.writerows(result)       
fp.close
