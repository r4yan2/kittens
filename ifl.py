#The Item-feature list is a structure that map the itemId with the list of feature in the item
import csv
from collections import defaultdict

byfeature = list(csv.reader(open('data/icm.csv', 'rb'), delimiter = ','))
del byfeature[0]
ifl = defaultdict(list)
lastitem=int(byfeature[0][0])
for elem in byfeature:
    elem = map (int, elem)
    ifl[elem[0]].append(elem[1])

# Putting result in a suitable format for writer
result=sorted(ifl.items(), key=lambda x:x[0])

# Writing Results
writer = (csv.writer(open('data/ifl.csv', 'w'), delimiter=','))
writer.writerow(["item","features"])
writer.writerows(result)
