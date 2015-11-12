import sys
import math
import csv
import time
import operator
import numpy as np
from scipy.spatial.distance import cosine
from collections import defaultdict

def cos(v1, v2):
    '''cosine similairty, hand implementation with numpy libraries'''
    return np.dot(v1, v2) / (np.sqrt(np.dot(v1, v1)) * np.sqrt(np.dot(v2, v2)))

def loadUserAvgRating():
    '''Populate the hashmap with the average given vote by all users'''
    count = {}
    count = defaultdict(lambda: 0,count)
    for elem in train:
        user=elem[0]
        evaluation=elem[2]
        avgUserRating[user] = (avgUserRating[user] * count[user] + float(evaluation)) / (count[user] + 1) # running average
        count[user] += 1

def getItemAvgRating():
    '''Populate the hashmap with the average taken vote by all items'''
    count = {}
    count = defaultdict(lambda: 0,count)
    for elem in train:
        item=elem[1]
        evaluation=elem[2]
        avgItemRating[item] = (avgItemRating[item] * count[item] + float(evaluation)) / (count[item] + 1) # running average
        count[item] += 1

def getUserExtendedEvaluationVector(u):
    '''return the 20k long vector with the user's rating at index corresponding to related item, 0 if no evaluation is provided'''
    evaluatedItems = getUserEvaluatedItems(u)
    userExtendedEvaluationVector = [0] * moviesNumber # initialization
    for (user,item),value in trainRddMappedValuesCollected:
        if (user == u and item in evaluatedItems):
            userExtendedEvaluationVector[item - 1] = value
    return userExtendedEvaluationVector

def getEvaluation(u,i):
    '''Getting the evaluation of a specific film for a user'''
    return ufvl[ (u,i) ]

def getRecommendetions(userToRecommend):
    '''
    getRecommendetions(user)
    take an user
    return the list of the recommendetions
    the method work of the vector of the seen items of the usrs. For all the
    users in the userSet we make similarity with the User for which we are
    recommending by:
    * taking the list of the common items
    * changind the items with the respective evaluation
    * removing already seen items
    * removing film not well evaluated
    * making a personaCorrelation between the two users to make the graduatory
    * TODO fill the description
    '''
    vectorUserToRecommend = getUserEvaluatedItems(userToRecommend) # get the vector of the seen items
    filmThreshold = xrange(7,10) # threshold to be applied to the possible Recommendetions
    similarities = {}
    possibleRecommendetions = []
    countForTheAverage=0
    avgCommonMovies = {}
    avgCommonMovies = defaultdict(lambda: 0,avgCommonMovies)

    for userIterator in userSet:
        skip = True # if no common film will be found this variable will remain setten and the remain part of the cicle will be skipped
        if (userIterator == userToRecommend): # if the user which we need to make recommendetion is the same of the one in the iterator we skip this iteration
            continue

        vectorUserIterator = getUserEvaluatedItems(userIterator)[:] #same as befor, this time we need to get a copy of the vector (achieved through [:]) since we are going to modify it
        evaluationListUserToRecommend = []
        evaluationListIteratorUser = []
        for filmAlreadySeen in list(vectorUserToRecommend):
            if filmAlreadySeen in vectorUserIterator and filmAlreadySeen in filmThreshold:
                skip = False
                evaluationListUserToRecommend.append(getEvaluation(UserToRecommend,filmAlreadySeen))
                evaluationListIteratorUser.append(getEvaluation(userIterator,filmAlreadySeen))
                vectorUserIterator.remove(filmAlreadySeen)
        avgCommonMovies[userIterator] = (avgCommonMovies[userIterator]  *  countForTheAverage + len(evaluationListIteratorUser)) / (countForTheAverage + 1) # Running Average
        countForTheAverage += 1
        if (skip):
            similarity = 0
        else:
            similarity = pearsonCorrelation(userToRecommend, userIterator, evaluationListUserToRecommend, evaluationListIteratorUser)
        if similarity > 0: # taking into consideation only positive similarities
            similarities[userIterator] = similarity
            movies.extend(vectorUserIterator) # extend needed to "append" a list without creating nesting
    possibleRecommendetions = set(possibleRecommendetions) # we need every element to be unique
    return getPredictions(userToRecommend, similarities, possibleRecommendetions) # functional python fuck yeah

def getPredictions(u, similarities, movies):
    '''This method is making the predictions for a given user'''
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

def getTopN():
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

def loadUserStats(): # TODO ufc is a memory leak, need a refactor
    '''Parsing the item feature list'''
    byfeature = list(csv.reader(open('data/icm.csv', 'rb'), delimiter = ','))
    del byfeature[0]
    for elem in byfeature:
        elem = map (int, elem)
        if not (elem[0]) in ifl:
            ifl[elem[0]] = []
        ifl[elem[0]].append(elem[1])

    '''Creating the UserEvaluatedList: the list of items already evaluated by an user'''
    for elem in train:
        elem = map(int, elem)
        u = elem[0]
        i = elem[1]
        r = elem[2]
        uel[u].append(i)
        urc[u] = urc[u] + 1
        if not i in ifl:
            continue
        for f in ifl[i]:
            ufr[(u,f)] = (ufr[(u,f)] + float(r)) / 2
            ufc[(u,f)] = ufc[(u,f)] + 1

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

def padding(u): # recicle from the old recommendetions methos
    personalizedTopN = {}
    topN=getTopN() # getter for the generic TopN
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
    return len(train)

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

def userItemEvaluationLoader():
    '''return an hashmap structured
    (K1,K2): V
    (user,film): evaluation
    this map is obtained mapping the correct field from train set into an hashmap'''
    return dict((((x[0], x[1]), x[2])) for x in map(lambda x: map(int,x),train[1:]))

def main():
    '''
    main loop
    for all the users in userSet make the recommendetions through getRecommendetions, the output of the function
    is properly sorted only the first top5 elements are considered, eventually it's possible to get padding for the
    user which getRecommendetions is unable to fill
    Includes also percentage and temporization
    '''
    resultToWrite=[]
    loopTime = time.time()
    for user in userSet:
        print "Completion percentage %f, increment %f" % (float(userSet.index(user)*100) / len(userSet),time.time() - loopTime)
        loopTime = time.time()
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
        resultToWrite.append(elem)
    resultWriter(resultToWrite)



'''List of Global Variables Definitions and Computation
    for debugging reason are out of the main function'''

trainRdd = trainRddLoader()

trainRddMappedValuesCollected=trainRdd.map(lambda x: map(int,x)).map(lambda x: (list((x[0],x[1])),x[2])).collect()
nUsers=getNumUsers()
train = trainLoader()
userItemEvaluationMap = userItemEvaluationLoader()
userSet = userSetLoader()
avgUserRating = {}
avgUserRating = defaultdict(lambda: 0.0,avgUserRating)
getUserAvgRating()

avgItemRating = {}
avgItemRating = defaultdict(lambda: 0.0,avgItemRating)
getItemAvgRating()

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
