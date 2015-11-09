import sys
import math
import csv
import time
import operator
from scipy.spatial.distance import cosine
import numpy as np
from collections import defaultdict

def cos(v1, v2):
    '''cosine similarity implemented in numpy'''
    return np.dot(v1, v2) / (np.sqrt(np.dot(v1, v1)) * np.sqrt(np.dot(v2, v2)))

def getUserAvgRating():
    '''User avg rating'''
    count={}
    count = defaultdict(lambda: 0,count)
    for elem in train:
        u=elem[0]
        v=elem[2]
        avgRating[u] = (avgRating[u]*count[u]+float(v))/(count[u]+1)
        count[u]=count[u]+1

def getUserHistoryExtendedVector(u):
    '''Creating the function getUserVector that return the 20k long vector with
    the user's rating for all items'''
    listItems = listUserItems(u)
    itemsList = [0]*moviesNumber
    for (user,item),value in trainRddMappedValuesCollected:
        if (user!=u or item not in listItems):
            continue
        itemsList[item-1]=value
    return itemsList

def getEvaluation(u,i):
    '''Getting the evaluation of a specific film for a user'''
    return ufvl[(u,i)]

def getRecommendetions(u):
    '''
    getRecommendetions(user)
    work on the user vector to make the cosine similarity
    if is greater than a threshold then the evaluation
    of the film is recorded in the possible recommendetions
    multiplied by the similarity value for that specific user
    '''
    v1 = listUserItems(u)
    filmThresh = xrange(6,10)
    recommendetions = []
    similarities = []

    for u2 in userSet:
        skip = True
        if (u2 == u):
            continue
        v2 = listUserItems(u2)
        listA = []
        listB = []
        for i in list(v1):
            if i in v2:
                skip = False
                listA.append(getEvaluation(u,i))
                listB.append(getEvaluation(u2,i))
                v2.remove(i)
        if (skip):
            similarity = 0
        else:
            similarity = 1.0 - pearsonCorrelation(u, u2, listA, listB)

        similarities.append(similarity)
        prediction(u, similarity)
    return recommendetions

def get_topN():
    # Insert into an hashmap the total value for each film calculated by summing all the rating obtained throught user rating divided by the sum of the votes + the variable shrink value obtained as logarithm of the number of votes divided for the number of users in the system.
    sum = 0
    counter = 0
    total = {}
    lastid = int(train[0][1])
    for line in train:
        line = map (int, line)
        if (lastid!=line[1]):
            variableShrink = int(math.fabs(math.log(float(counter)/nUsers)))
            total[lastid] = sum/float(counter + variableShrink);
            counter = 0
            sum = 0
            lastid = line[1]
        sum = sum + line[2]
        counter = counter + 1
    # Sorting in descending order the list of items
    return sorted(total.items(), key=lambda x:x[1], reverse=True)

    '''alternatively get it from the csv'''
    return dict(list(csv.reader(open('data/topN.csv', 'rb'), delimiter = ',')))

def loadUserStats():
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

    '''Creating the UserEvaluatedList: the list of items already evaluated by an user'''
    for line in train:
            line = map (int, line)
            uel[line[0]].append(line[1])
    '''Parsing the item feature list'''
    byfeature = list(csv.reader(open('data/icm.csv', 'rb'), delimiter = ','))
    del byfeature[0]
    for elem in byfeature:
        elem = map (int, elem)
        if not (elem[0]) in ifl:
            ifl[elem[0]]=[]
        ifl[elem[0]].append(elem[1])


def listUserItems(user):
    '''List of user's seen items'''
    return itemsList[usersList.index(user)][:]

def numDistinctItems():
    ''' Getting the number of items that we have in our icm.csv'''
    icmRdd = sc.textFile("data/icm.csv").map(lambda line: line.split(","))
    icmFirstRow = icmRdd.first()
    icmRdd=icmRdd.filter(lambda x:x !=icmFirstRow)
    numDistinctItems = icmRdd.map(lambda x: map(int,x)).sortByKey(False).keys().first()
    return numDistinctItems

def padding(u):
    personalizedTopN={}
    for i in dictTopN5:
        i=int(i)
        personalizedTopN[(u,i)]=math.log(float(dictTopN5[str(i)]))
        if not i in ifl:
            continue
        for f in ifl[i]:
            if not (ufc[(u,f)]==0 or urc[u]==0 or u not in urc or (u,f) not in ufc):
                personalizedTopN[(u,i)]=personalizedTopN[(u,i)]+ufr[(u,f)]/(ufc[(u,f)]/urc[u])
    topNPersonalized=sorted(personalizedTopN.items(), key=lambda x:x[1], reverse=True)
    count=0
    iterator=0
    recommendetions=''
    while count<5:
        if (topNPersonalized[iterator][0][0]==u):
            if not (topNPersonalized[iterator][0][1] in uel[u]):
                recommendetions=recommendetions+(str(topNPersonalized[iterator][0][1])+' ')
                count=count+1
        iterator=iterator+1
    return recommendetions

def pearsonCorrelation(u, u2, listA, listB):
    '''Calculating the Pearson Correlation coefficient between two given users'''
    avgu = avgRating[u]
    avgu2 = avgRating[u2]
    for item1, item2 in zip(listA, listB):
        listNumeratorU = map(lambda listA: listA-avgu, listA)
        listNumeratorU2 = map(lambda listB: listB-avgu2, listB)
        numeratorPearson = np.sum([elem1*elem2 for elem1,elem2 in zip(listNumeratorU,listNumeratorU2)])
        listDenU = map(lambda listNumeratorU: listNumeratorU**2, listNumeratorU)
        listDenU2 = map(lambda listNumeratorU2: listNumeratorU2**2, listNumeratorU2)
        denominatorPearson = math.sqrt(np.sum(listDenU))*math.sqrt(np.sum(listDenU2))
        pearson = numeratorPearson/denominatorPearson
    return pearson

def resultWriter(result):
    ''' Writing Results '''
    with open ('data/result.csv', 'w') as fp:
        writer = csv.writer(fp, delimiter=',')
        writer.writerow(["userId,testItems"])
        writer.writerows(result)
    fp.close

def getNumUsers():
    '''Calculating the number of distinct users that rated some items'''
    return trainRddLoader().map(lambda r: r[0]).distinct().count()

def getNumRatings():
    '''Calculating the number of ratings that of the items'''
    return trainRddLoader().count()

def getNumMovies():
    '''Calculating the number of items that some users rated them'''
    numMovies = trainRddLoader().map(lambda r: r[1]).distinct().count()

def trainLoader():
    train = list (csv.reader(open('data/train.csv', 'rb'), delimiter = ','))
    del train[0] # deletion of the string header
    trainMapped = []
    for i in train:
        trainMapped.append(map(int,i))
    return trainMapped

def trainRddLoader():
    '''Creating the RDD of the train.csv'''
    trainRdd = sc.textFile("data/train.csv").map(lambda line: line.split(","))
    '''Getting the header of trainRdd'''
    trainFirstRow = trainRdd.first()
    '''Removing the first line of train, that is userId,iteamId,rating'''
    trainRdd=trainRdd.filter(lambda x:x !=trainFirstRow)
    return trainRdd

def userSetLoader():
    '''Get of userSet'''
    userSet = sc.textFile("data/test.csv").map(lambda line: line.split(","))
    userSetFirstRow = userSet.first()
    userSet = userSet.filter(lambda x:x !=userSetFirstRow).keys().map(lambda x: int(x)).collect()
    return userSet

def ufvlLoader():
    return dict((((x[0], x[1]), x[2])) for x in map(lambda x: map(int,x),train[1:]))

def main():
    '''
    main loop
    for all the users in userSet make the recommendetions through getRecommendetions, the output of the function
    is properly sorted onyl the first top5 elements are considered, eventually it's possible to get padding for the
    user which getRecommendetions is unable to fill
    Includes also percentage and temporization
    '''
    loop_time=time.time()
    for user in userSet:
        #sys.stdout.flush()
        #sys.stdout.write("\r %s % %s" % (str(float(userSet.index(user)*100)/len(userSet)),str(time.time()-loop_time)))
        print "Completion percentage %f, increment %f" % (float(userSet.index(user)*100)/len(userSet),time.time()-loop_time)
        loop_time=time.time()
        recommend=''
        recommendetions=sorted(getRecommendetions(user), key=lambda x:x[1], reverse=True)[:5]
        for i,v in recommendetions:
            recommend=recommend+(str(i)+' ')
        #if (len(recommendetions)<5):
        #    recommend=padding(user)
        elem=[]
        elem.append(user)
        elem.append(recommend)
        result.append(elem)
    resultWriter(result)



'''List of Global Variables Definitions and Computation'''

trainRdd = trainRddLoader()

itemsList = trainRdd.map(lambda x: [x[0],x[1]]).groupByKey().map(lambda x: (int(x[0]), map(int,list(x[1])))).values().collect()
usersList = trainRdd.map(lambda x: [x[0],x[1]]).groupByKey().map(lambda x: (int(x[0]), map(int,list(x[1])))).keys().collect()
trainRddMappedValuesCollected=trainRdd.map(lambda x: map(int,x)).map(lambda x: (list((x[0],x[1])),x[2])).collect()
nUsers=getNumUsers()
train = trainLoader()
ufvl = ufvlLoader()
userSet = userSetLoader()
avgRating = {}
avgRating = defaultdict(lambda: 0.0,avgRating)
getUserAvgRating()
ufr = {}
moviesNumber = numDistinctItems()
ufc = {}
urc = {}
dictTopN5 = {}
ufc = defaultdict(lambda: 0.0, ufc)
ufr = defaultdict(lambda: 0.0, ufr)
urc = defaultdict(lambda: 0,urc)
uel = defaultdict(list)
ifl = {}
loadUserStats()

result=[]
