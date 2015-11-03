import csv
import math
import operator
from collections import defaultdict

# Parsing train dataser
train = list (csv.reader(open('data/train.csv', 'rb'), delimiter = ','))
del train[0] #deletion of the string header

# Creating the UserEvaluatedList: the list of items already evaluated by an user
uel = defaultdict(list)
for line in train:
        line = map (int, line)
        uel[line[0]].append(line[1])

result=sorted(uel.items(), key=lambda x:x[0])

# Writing Results
with open ('data/uel.csv', 'w') as fp:
     writer = csv.writer(fp, delimiter=',')
     #writer.writerow(["userId,testItems"]) #header line
     writer.writerows(result)
fp.close


