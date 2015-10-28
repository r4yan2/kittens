import csv
import math
import operator
from collections import defaultdict

# Parsing the train.csv
train = list (csv.reader(open('train.csv', 'rb'), delimiter = ','))

# Writing Results
writer = (csv.writer(open('result.csv', 'w'), delimiter=','))

# Calculating number of users

nUsers=len(train)

# Insert into an hashmap the shrink value associated to an item and another hashmap with the normalized rating
writer = (csv.writer(open('shrinkage.csv', 'w'), delimiter=','))
writer.writerow(['item','Shrink value'])
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
shrinklist=sorted(shrink.items(), key=lambda x:x[0])
writer.writerows(shrinklist)
