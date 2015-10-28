import csv
import math
import operator
from collections import defaultdict

# Parsing train dataser
train = list (csv.reader(open('data/train.csv', 'rb'), delimiter = ','))
del train[0] #deletion of the string header

# Parsing test dataset
test = list (csv.reader(open('data/test.csv', 'rb'), delimiter = ','))
del test[0]

# initializing resut array
result=[]

# Writing Results
writer = (csv.writer(open('data/result.csv', 'w'), delimiter=','))
writer.writerow("userId","testItems")

# Calculating number of users
nUsers=len(train)

### TODO: parse the topN.csv

# Creating the UserEvaluatedList: the list of items already evaluated by an user
uel = defaultdict(list)
for line in train:
        line = map (int, line)
        uel[line[0]].append(line[1])
 
# Parsing the item feature list
byfeature = list(csv.reader(open('data/icm.csv', 'rb'), delimiter = ','))
del byfeature[0]
ifl = defaultdict(list)
lastitem=int(byfeature[0][0])
for elem in byfeature:
    elem = map (int, elem)
    ifl[elem[0]].append(elem[1])

# User feature avg rating
ufr={}
ufc={}
ufc = defaultdict(lambda: 0.0, ufc)
ufr = defaultdict(lambda: 0.0, ufr)
for i in train:
    i=map(int, i)
    for j in ifl[i[1]]:
        ufr[(i[0],j)]=(ufr[(i[0],j)]+float(i[2]))/2
        ufc[j]=ufc[j]+1 

## TODO fare la top-N personalizzata con la normalizzazione del voto basata sulle medie valutazioni dell'utente per una certa feature

personalizedTopN={}
dictTopN5=dict(topN[:3000])
for user in test:
    for elem in dictTopN5:
        personalizedTopN[(int(user[0]),elem)]=dictTopN5[elem]
        for i in ifl[elem]:
            personalizedTopN[(int(user[0]),elem)]=personalizedTopN[(int(user[0]),elem)]+ufr[(int(user[0]),i)]-5-math.fabs(math.log(math.fabs((ufr[(int(user[0]),i)]+0.01)/ufc[i])))

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

writer.writerows(result)    

