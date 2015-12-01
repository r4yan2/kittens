import math
import csv

# Parsing train dataser
train = list (csv.reader(open('data/train.csv', 'rb'), delimiter = ','))
del train[0] #deletion of the string header

# Parsing test dataset
test = list (csv.reader(open('data/test.csv', 'rb'), delimiter = ','))
del test[0]

# Calculating number of users
nUsers=len(test)

# Insert into an hashmap the total value for each film calculated by summing all the rating obtained throught user rating divided by the sum of the votes + the variable shrink value obtained as logarithm of the number of votes divided for the number of users in the system.
sum=0
counter=0
total={}
lastid=int(train[0][1])
for line in train:
    line = map (int, line)
    if (lastid!=line[1]):
        total[lastid]=sum/float(counter+int(math.fabs(math.log(float(counter)/nUsers))));
        counter=0;
        sum=0;
        lastid=line[1];
    sum=sum+line[2]
    counter=counter+1

# Sorting in descending order the list of items
top_n=sorted(total.items(), key=lambda x:x[1], reverse=True)

# Writing Results
writer = (csv.writer(open('data/topN.csv', 'w'), delimiter=','))
writer.writerow(["Items", "value"])
writer.writerows(top_n)
