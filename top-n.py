import csv
import math
import operator
from collections import defaultdict

## Parsing the train.csv
parsed = list (csv.reader(open('train.csv', 'rb'), delimiter = ','))

## Calculating number of users

nUsers=len(parsed)

## Insert into an hashmap the shrink vaue associated to an item and another hashmap with the normalized rating 
sum=0
counter=0
total={}
shrink={}
del parsed[0]
lastid=int(parsed[0][1])
for line in parsed:
    line = map (int, line)
    if (lastid!=line[1]):
	shrink[lastid]=int(math.fabs(math.log(float(counter)/nUsers)));
        total[lastid]=sum/float(counter+shrink[lastid]);
        counter=0;
        sum=0;
        lastid=line[1];
    sum=sum+line[2]
    counter=counter+1

## Sorting in descending order the list of items
sorted_total = dict(sorted(total.items(), key=operator.itemgetter(1), reverse=True))
top=sorted(sorted_total.items(), key=lambda x:x[1], reverse=True)

## Creating the UserEvaluatedList that is the list of items already evaluated by an user
uel = defaultdict(list)
for line in parsed:
        line = map (int, line)
        uel[line[0]].append(line[1])
 
## Creating results by filtering out of the top list the items already seen by users 
submission = list (csv.reader(open('test.csv', 'rb'), delimiter = ','))
test=[]
test.append(['userId','testItems'])
del submission[0]
for elem in submission:
    count=0
    iterator=0
    recommendetions=''
    while count<5:
        if not (top[iterator][0] in uel[int(elem[0])]):
            recommendetions=recommendetions+(str(top[iterator][0])+' ')
            iterator=iterator+1
            count=count+1
        else:
            iterator=iterator+1
    elem.append(recommendetions)
    test.append(elem)
## Creating the item feature map, that is the hashmap containing all the items associated with the features they have
ifl = defaultdict(list)
lastitem=int(byfeature[0][0])
for elem in byfeature:
    elem = map (int, elem)
    ifl[elem[0]].append(elem[1])


## Writing Results
with open ('test.csv', 'w') as fp:
     a = csv.writer(fp, delimiter=',')
     a.writerows(test)       
fp.close
