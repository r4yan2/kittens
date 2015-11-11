import sys
import math
import csv
import time
import operator
import numpy as np
from scipy.spatial.distance import cosine
from collections import defaultdict

def cos(v1, v2):
    '''cosine similarity implemented in numpy'''
    return np.dot(v1, v2) / (np.sqrt(np.dot(v1, v1)) * np.sqrt(np.dot(v2, v2)))

def getUserAvgRating():
    '''User avg rating'''
    count = {}
    count = defaultdict(lambda: 0,count)
    for elem in train:
        u=elem[0]
        v=elem[2]
        avgUserRating[u] = (avgUserRating[u] * count[u] + float(v)) / (count[u] + 1)
        count[u] = count[u] + 1

def getItemAvgRating():
    '''Item avg rating'''
    count = {}
    count = defaultdict(lambda: 0,count)
    for elem in train:
        i=elem[1]
        v=elem[2]
        avgItemRating[i] = (avgItemRating[i] * count[i] + float(v)) / (count[i] + 1)
        count[i] = count[i] + 1

def getUserHistoryExtendedVector(u):
    '''Creating the function getUserVector that return the 20k long vector with
    the user's rating for all items'''
    listItems = getUserEvaluatedItems(u)
    itemsList = [0] * moviesNumber
    for (user,item),value in trainRddMappedValuesCollected:
        if (user != u or item not in listItems):
            continue
        itemsList[item - 1] = value
    return itemsList

def getEvaluation(u,i):
    '''Getting the evaluation of a specific film for a user'''
    return ufvl[ (u,i) ]

def getRecommendetions(u):
    '''
    getRecommendetions(user)
    works on the user vector to make the similarity
    if it is greater than a threshold then the evaluation
    of the film is recorded in the possible recommendetions
    multiplied by the similarity value for that specific user
    '''
    v1 = getUserEvaluatedItems(u)
    filmThresh = xrange(6,10)
    recommendetions = []
    similarities = {}
    movies = []
    for u2 in userSet:
        skip = True
        if (u2 == u):
            continue
        v2 = getUserEvaluatedItems(u2)[:]
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
            similarity = pearsonCorrelation(u, u2, listA, listB)
        if similarity > 0:
            similarities[u2] = similarity
            movies.extend(v2)
    movies = set(movies)
    return getPredictions(u, similarities, movies)

'''This method is making the predictions for a given user'''
def getPredictions(u, similarities, movies):
    avgu = avgUserRating[u]
    userValues = []
    predictions = []
    denominator = np.sum(similarities.values())
    for item in movies:
        for u2 in similarities.keys():
            if item in getUserEvaluatedItems(u2):
                avg2 = avgUserRating[u2]
                rating = getEvaluation(u2,item)
                userValues.append(similarities[u2] * (rating - avg2))
        numerator = np.sum(userValues)
        prediction = avgu + float(numerator)/denominator
        predictions.append( (item,prediction) )
    return predictions

def get_topN():
    # Insert into an hashmap the total value for each film calculated by summing all the rating obtained throught user rating divided by the sum of the votes + the variable shrink value obtained as logarithm of the number of votes divided for the number of users in the system.
    sum = 0
    counter = 0
    total = {}
    lastid = int(train[0][1])
    for line in train:
        line = map (int, line)
        if (lastid != line[1]):
            variableShrink = int(math.fabs(math.log(float(counter) / nUsers)))
            total[lastid] = sum / float(counter + variableShrink);
            counter = 0
            sum = 0
            lastid = line[1]
        sum = sum + line[2]
        counter = counter + 1
    # Sorting in descending order the list of items
    return sorted(total.items(), key = lambda x:x[1], reverse = True)

    '''alternatively get it from the csv'''
    return dict(list(csv.reader(open('data/topN.csv', 'rb'), delimiter = ',')))

def loadUserStats():
    '''Parsing the item feature list'''
    byfeature = list(csv.reader(open('data/icm.csv', 'rb'), delimiter = ','))
    del byfeature[0]
    for elem in byfeature:
        elem = map (int, elem)
        if not (elem[0]) in ifl:
            ifl[elem[0]] = []
        ifl[elem[0]].append(elem[1])

    for elem in train:
        elem = map(int, elem)
        u = elem[0]
        i = elem[1]
        r = elem[2]
        urc[u] = urc[u] + 1
        if not i in ifl:
            continue
        for f in ifl[i]:
            ufr[(u,f)] = (ufr[(u,f)] + float(r)) / 2
            ufc[(u,f)] = ufc[(u,f)] + 1

    '''Creating the UserEvaluatedList: the list of items already evaluated by an user'''
    for line in train:
            line = map (int, line)
            uel[line[0]].append(line[1])

def getUserEvaluatedItems(user):
    '''List of user's seen items'''
    return uel[user]

def numDistinctItems():
    ''' Getting the number of items that we have in our icm.csv'''
    icmRdd = sc.textFile("data/icm.csv").map(lambda line: line.split(","))
    icmFirstRow = icmRdd.first()
    icmRdd = icmRdd.filter(lambda x:x != icmFirstRow)
    numDistinctItems = icmRdd.map(lambda x: map(int,x)).sortByKey(False).keys().first()
    return numDistinctItems

def padding(u):
    personalizedTopN = {}
    topN=get_topN()
    for i,v in topN:
        personalizedTopN[i] = math.log(v)
        if not i in ifl:
            continue
        for f in ifl[i]:
            if not (ufc[(u,f)] == 0 or urc[u] == 0 or u not in urc or (u,f) not in ufc):
                personalizedTopN[i] = personalizedTopN[i] + ufr[(u,f)] / (ufc[(u,f)] / urc[u])
    topNPersonalized = sorted(personalizedTopN.items(), key = lambda x:x[1], reverse = True)
    count = 0
    iterator = 0
    recommendetions = ''
    while count<5:
        if not (topNPersonalized[iterator][0] in uel[u]):
            recommendetions = recommendetions + (str(topNPersonalized[iterator][0]) + ' ')
            count = count + 1
        iterator = iterator + 1
    return recommendetions

def pearsonCorrelation(u, u2, listA, listB):
    '''Calculating the Pearson Correlation coefficient between two given users'''
    avgu = avgUserRating[u]
    avgu2 = avgUserRating[u2]
    for item1, item2 in zip(listA, listB):
        listNumeratorU = map(lambda listA: listA - avgu, listA)
        listNumeratorU2 = map(lambda listB: listB - avgu2, listB)
        numeratorPearson = np.sum([elem1*elem2 for elem1,elem2 in zip(listNumeratorU,listNumeratorU2)])
        listDenU = map(lambda listNumeratorU: listNumeratorU**2, listNumeratorU)
        listDenU2 = map(lambda listNumeratorU2: listNumeratorU2**2, listNumeratorU2)
        denominatorPearson = math.sqrt(np.sum(listDenU))*math.sqrt(np.sum(listDenU2))
        if denominatorPearson==0:
            return 0
        pearson = numeratorPearson/denominatorPearson
    return pearson

def resultWriter(result):
    ''' Writing Results '''
    with open ('data/result.csv', 'w') as fp:
        writer = csv.writer(fp, delimiter = ',')
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
    trainRdd = trainRdd.filter(lambda x:x != trainFirstRow)
    return trainRdd

def userSetLoader():
    '''Get of userSet'''
    userSet = sc.textFile("data/test.csv").map(lambda line: line.split(","))
    userSetFirstRow = userSet.first()
    userSet = userSet.filter(lambda x:x != userSetFirstRow).keys().map(lambda x: int(x)).collect()
    return userSet

def ufvlLoader():
    return dict((((x[0], x[1]), x[2])) for x in map(lambda x: map(int,x),train[1:]))

def main():
    '''
    main loop
    for all the users in userSet make the recommendetions through getRecommendetions, the output of the function
    is properly sorted only the first top5 elements are considered, eventually it's possible to get padding for the
    user which getRecommendetions is unable to fill
    Includes also percentage and temporization
    '''
    result=[]
    loop_time = time.time()
    for user in userSet:
        #sys.stdout.flush()
        #sys.stdout.write("\r %s % %s" % (str(float(userSet.index(user)*100)/len(userSet)),str(time.time()-loop_time)))
        print "Completion percentage %f, increment %f" % (float(userSet.index(user)*100) / len(userSet),time.time() - loop_time)
        loop_time = time.time()
        recommend = ''
        recommendetions = sorted(getRecommendetions(user), key = lambda x:x[1], reverse=True)[:5]
        print user
        for i,v in recommendetions:
            recommend = recommend+(str(i) + ' ')
        if (len(recommendetions)<5):
            recommend=padding(user)
        print recommend
        elem = []
        elem.append(user)
        elem.append(recommend)
        result.append(elem)
    resultWriter(result)



'''List of Global Variables Definitions and Computation'''

trainRdd = trainRddLoader()

trainRddMappedValuesCollected=trainRdd.map(lambda x: map(int,x)).map(lambda x: (list((x[0],x[1])),x[2])).collect()
nUsers=getNumUsers()
train = trainLoader()
ufvl = ufvlLoader()
userSet = userSetLoader()
avgUserRating = {}
avgUserRating = defaultdict(lambda: 0.0,avgUserRating)
getUserAvgRating()
ufr = {}
moviesNumber = numDistinctItems()
ufc = {}
urc = {}
ufc = defaultdict(lambda: 0.0, ufc)
ufr = defaultdict(lambda: 0.0, ufr)
urc = defaultdict(lambda: 0,urc)
uel = defaultdict(list)
ifl = {}
loadUserStats()
