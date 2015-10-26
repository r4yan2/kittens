import csv
import math
import operator
from collections import defaultdict

# Parsing the train.csv
train = list (csv.reader(open('train.csv', 'rb'), delimiter = ','))

# Calculating number of users

nUsers=len(train)

# Insert into an hashmap the shrink value associated to an item and another hashmap with the normalized rating 
sum=0
counter=0
total={}
shrink={}
del train[0]
lastid=int(train[0][1])
for line in train:
    line = map (int, line)
    if (lastid!=line[1]):
	shrink[lastid]=int(math.fabs(math.log(float(counter)/nUsers)));
        total[lastid]=sum/float(counter+shrink[lastid]);
        counter=0;
        sum=0;
        lastid=line[1];
    sum=sum+line[2]
    counter=counter+1

# Sorting in descending order the list of items
sortedTotal = dict(sorted(total.items(), key=operator.itemgetter(1), reverse=True))
topN=sorted(sortedTotal.items(), key=lambda x:x[1], reverse=True)

# Creating the UserEvaluatedList that is the list of items already evaluated by an user

uel = defaultdict(list)
for line in train:
        line = map (int, line)
        uel[line[0]].append(line[1])
 
# Creating results by filtering out of the top list the items already seen by users
 
test = list (csv.reader(open('test.csv', 'rb'), delimiter = ','))
result = []
result.append(['userId','testItems'])
del test[0]
for elem in test:
    count=0
    iterator=0
    recommendetions=''
    while count<5:
        if not (topN[iterator][0] in uel[int(elem[0])]):
            recommendetions=recommendetions+(str(topN[iterator][0])+' ')
            iterator=iterator+1
            count=count+1
        else:
            iterator=iterator+1
    elem.append(recommendetions)
    result.append(elem)

## Creating the item feature map, that is the hashmap containing all the items associated with the features they have

byfeature = list(csv.reader(open('icm.csv', 'rb'), delimiter = ','))
del byfeature[0]
ifl = defaultdict(list)
lastitem=int(byfeature[0][0])
for elem in byfeature:
    elem = map (int, elem)
    ifl[elem[0]].append(elem[1])

## User feature avg rating
ufr={}
ufr = defaultdict(lambda: 0.0, ufr)
for i in train:
    i=map(int, i)
    for j in ifl[i[1]]:
        ufr[(i[0],j)]=(ufr[(i[0],j)]+float(i[2]))/2

## TODO fare la top-N personalizzata con la normalizzazione del voto basata sulle medie valutazioni dell'utente per una certa feature

personalizedTopN={}
for user in test:
    for elem in total:
	personalizedTopN[(int(user[0]),elem)]=total[elem]
        for i in ifl[elem]:
            personalizedTopN[(int(user[0]),elem)]=personalizedTopN[(int(user[0]),elem)]+ufr[int(user[0]),i]-5

sortedPersonalized = dict(sorted(personalizedTopN.items(), key=operator.itemgetter(2), reverse=True))
topNPersonalized=sorted(sortedPersonalized.items(), key=lambda x:x[2], reverse=True)

test = list (csv.reader(open('test.csv', 'rb'), delimiter = ','))
result = []
result.append(['userId','testItems'])
del test[0]
for elem in test:
    count=0
    iterator=0
    recommendetions=''
    while count<5:
        if not (topNPersonalized[(int(elem[0]),iterator)][0] in uel[int(elem[0])]):
            recommendetions=recommendetions+(str(topNPersonalized[(int(elem[0]),iterator)][0])+' ')
            iterator=iterator+1
            count=count+1
        else:
            iterator=iterator+1
    elem.append(recommendetions)
    result.append(elem)


## Writing Results
with open ('test.csv', 'w') as fp:
     a = csv.writer(fp, delimiter=',')
     a.writerows(result)       
fp.close
