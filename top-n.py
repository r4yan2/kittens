import csv
import operator

parsed = list (csv.reader(open('train.csv', 'rb'), delimiter = ','))
sum=0
counter=0
total={}
shrink=3
del parsed[0]
lastid=parsed[0][1]
for line in parsed:
    line = map (int, line)
    if (lastid!=line[1]):
        total[lastid]=sum/float(counter+shrink);
        counter=0;
        sum=0;
        lastid=line[1];
    sum=sum+line[2]
    counter=counter+1

sorted_total = dict(sorted(total.items(), key=operator.itemgetter(1), reverse=True))
top=sorted(sorted_total.items(), key=lambda x:x[1], reverse=True)

from collections import defaultdict
uel = defaultdict(list)
for line in parsed:
        line = map (int, line)
        uel[line[0]].append(line[1])
        
submission = list (csv.reader(open('test.csv', 'rb'), delimiter = ','))
test=[]
del submission[0]
for elem in submission:
    count=0
    iterator=0
    recommendetions=''
    while count<5:
        if not (top[iterator][0] in uel[int(elem[0])]):
            #elem.append(top[iterator][0])
            recommendetions=recommendetions+(str(top[iterator][0])+' ')
            iterator=iterator+1
            count=count+1
        else:
            iterator=iterator+1
    elem.append(recommendetions)
    test.append(elem)
with open ('test.csv', 'w') as fp:
     a = csv.writer(fp, delimiter=',')
     a.writerows(test)
        
fp.close
