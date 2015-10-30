import csv
from collections import defaultdict
# Parsing train dataser
train = list (csv.reader(open('data/train.csv', 'rb'), delimiter = ','))
del train[0] #deletion of the string header

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

# Putting result in a suitable format for writer
ufr=sorted(ufr.items(), key=lambda x:x[0])
ufc=sorted(ufc.items(), key=lambda x:x[0])
urc=sorted(urc.items(), key=lambda x:x[0])


# Writing Results
writer = (csv.writer(open('data/ufr.csv', 'w'), delimiter=','))
writer.writerows(ufr)

writer = (csv.writer(open('data/ufc.csv', 'w'), delimiter=','))
writer.writerows(ufc)

writer = (csv.writer(open('data/urc.csv', 'w'), delimiter=','))
writer.writerows(urc)

#To remove the " character from the result type 'cat ifl.csv | tr -d '\"' > ifl.csv'

