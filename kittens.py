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
    global avgUserRating
    avgUserRating = {}
    count = {}
    count = defaultdict(lambda: 0,count)
    for elem in train:
        user=elem[0]
        evaluation=elem[2]
        try:
            avgUserRating[user] = (avgUserRating[user] * count[user] + float(evaluation)) / (count[user] + 1) # running average
        except Exception,e:
            avgUserRating[user] = 0.0
            avgUserRating[user] = (avgUserRating[user] * count[user] + float(evaluation)) / (count[user] + 1)
        count[user] += 1

def loadItemAvgRating():
    '''Populate the hashmap with the average taken vote by all items'''
    global avgItemRating
    avgItemRating = {}
    count = {}
    count = defaultdict(lambda: 0,count)
    for elem in train:
        item=elem[1]
        evaluation=elem[2]
        try:
            avgItemRating[item] = (avgItemRating[item] * count[item] + float(evaluation)) / (count[item] + 1) # running average
        except Exception,e:
            avgItemRating[item] = 0.0
            avgItemRating[item] = (avgItemRating[item] * count[item] + float(evaluation)) / (count[item] + 1) # running average
        count[item] += 1

def getUserExtendedEvaluationVector(u):
    '''return the 20k long vector with the user's rating at index corresponding to related item, 0 if no evaluation is provided'''
    evaluatedItems = getUserEvaluatedItems(u)
    userExtendedEvaluationVector = [0] * moviesNumber # initialization
    for (user,item),value in userEvaluationList:
        if (user == u and item in evaluatedItems):
            userExtendedEvaluationVector[item - 1] = value
    return userExtendedEvaluationVector

def getEvaluation(u,i):
    '''Getting the evaluation of a specific film for a user'''
    return userItemEvaluation[ (u,i) ]

def getRecommendetions(user):
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
    itemsUser = getUserEvaluationList(user) # get the vector of the seen items
    threshold = xrange(7,10) # threshold to be applied to the possible Recommendetions
    similarities = {}
    countFeature = {}
    countFeature = defaultdict(lambda: 0,countFeature)
    countForTheAverage = {}
    countForTheAverage = defaultdict(lambda: 0,countForTheAverage)
    avgCommonMovies = {}
    avgCommonMovies = defaultdict(lambda: 0.0,avgCommonMovies)
    numberCommonMovies = {}
    numberCommonMovies = defaultdict(lambda: 0,numberCommonMovies)
    possibleRecommendetions = {}
    possibleRecommendetions = defaultdict(list)
    evaluationsList = {}
    evaluationsList = defaultdict(list)
    blacklist=[]
    featuresAvg = {}
    featuresAvg = defaultdict(lambda: 0, featuresAvg)

    for userIterator in userSet:
        skip = True # if no common film will be found this variable will remain setten and the remain part of the cicle will be skipped
        if (userIterator == user): # if the user which we need to make recommendetion is the same of the one in the iterator we skip this iteration
            blacklist.append(userIterator)
            continue
        itemsUserIterator = getUserEvaluationList(userIterator)[:] #same as before, this time we need to get a copy of the vector (achieved through [:]) since we are going to modify it
        ratingsUser = [] #will contain the evaluations of User of the common items with userIterator
        ratingsUserIterator = [] #will contain the evaluations of userIterato of the common items with userIterator
        for item in list(itemsUser):
            if item in itemsUserIterator:
                numberCommonMovies[userIterator] += 1
                skip = False
                ratingsUser.append(getEvaluation(user,item))
                ratingsUserIterator.append(getEvaluation(userIterator,item))
                itemsUserIterator.remove(item)
        for item in itemsUserIterator:
            features = getFeaturesList(item)
            for feature in features:
                rating = getUserFeatureEvaluation(user,feature)
                featuresAvg[item] = (featuresAvg[item] * countFeature[item] + float(rating)) / (countFeature[item] + 1)
                countFeature[item] = countFeature[item] + 1
            if featuresAvg[item] in threshold:
                possibleRecommendetions[userIterator].append( item )

        if (skip or len(possibleRecommendetions)==0) :
            blacklist.append(userIterator)
            continue

        evaluationsList[userIterator].append(ratingsUser)
        evaluationsList[userIterator].append(ratingsUserIterator)

        avgCommonMovies[userIterator] = (avgCommonMovies[userIterator]  *  countForTheAverage[userIterator] + len(ratingsUserIterator)) / (countForTheAverage[userIterator] + 1) # Running Average
        countForTheAverage[userIterator] += 1

    for userIterator in userSet:
        if (userIterator in blacklist):
            continue
        if numberCommonMovies[userIterator] >= avgCommonMovies[userIterator]:
            similarity = pearsonCorrelation(user, userIterator, evaluationsList[userIterator][0], evaluationsList[userIterator][1])
        else:
            similarity = pearsonCorrelation(user, userIterator, evaluationsList[userIterator][0], evaluationsList[userIterator][1]) * (numberCommonMovies[userIterator]/avgCommonMovies[userIterator]) # significance weight

        if similarity > 0.5: # taking into consideration only positive and significant similarities
            similarities[userIterator] = similarity

    return getPredictions(user, similarities, possibleRecommendetions) # we need every element to be unique

def getPredictions(user, similarities, possibleRecommendetions):
    '''This method is making the predictions for a given user'''
    avgu = avgUserRating[user]
    userValues = []
    predictions = []
    denominator = np.sum(similarities.values())
    for userIterator in similarities.keys():
        for item in possibleRecommendetions[userIterator]:
            avg2 = avgUserRating[userIterator]
            rating = getEvaluation(userIterator,item)
            userValues.append(similarities[userIterator] * (rating - avg2))
        numerator = np.sum(userValues)
        prediction = avgu + float(numerator)/denominator
        predictions.append( (item,prediction) )
    return predictions

def loadTopN():
    # Insert into an hashmap the total value for each film calculated by summing all the rating obtained throught user rating divided by the sum of the votes + the variable shrink value obtained as logarithm of the number of votes divided for the number of users in the system.
    global topN
    sum = 0
    counter = 0
    total = {}
    lastid = int(train[0][1])
    for line in train:
        line = map (int, line)
        if (lastid != line[1]):
            variableShrink = int(math.fabs(math.log(float(counter) / getNumUsers())))
            total[lastid] = sum / float(counter + variableShrink);
            counter = 0
            sum = 0
            lastid = line[1]
        sum = sum + line[2]
        counter = counter + 1
    # Sorting in descending order the list of items
    topN = sorted(total.items(), key = lambda x:x[1], reverse = True)

def loadMaps():
    '''Parsing the item feature list'''
    byfeature = list(csv.reader(open('data/icm.csv', 'rb'), delimiter = ',')) #open csv splitting field on the comma character
    del byfeature[0] # header remove
    global itemFeatureslist
    itemFeatureslist = {}
    for elem in byfeature:
        elem = map (int, elem)
        if not (elem[0]) in itemFeatureslist:
            itemFeatureslist[elem[0]] = []
        itemFeatureslist[elem[0]].append(elem[1])

    '''Creating some maps
    userEvaluationList: the list of items already evaluated by an user
    userEvaluationCount: not needed anymore, instead use len(userEvaluationList)
    userFeatureEvaluation: the map of the rating given to a feature calculated as average of the votes received through film containing that feature
    userFeatureEvaluationCount: count of the rating given by a user to a film containing that feature
    '''
    global userFeatureEvaluation # define variable as global
    userFeatureEvaluation = {}
    global userFeatureEvaluationCount # define variable as global
    userFeatureEvaluationCount = {}
    global userEvaluationList # define variable as global
    userEvaluationList = {}

    for elem in train:
        elem = map(int, elem)
        u = elem[0]
        i = elem[1]
        r = elem[2]

        setUserEvaluationList(u,i)

        if i in itemFeatureslist:
            for f in itemFeatureslist[i]:
                setUserFeatureEvaluationAndCount(u,f,r)

def getFeaturesList(i):
    try:
        return itemFeatureList[i]
    except Exception,e:
        return []

def setUserFeatureEvaluationAndCount(u,f,r):
    try: # need to manage the "initialization case" in which the key does not exists
        userFeatureEvaluation[(u,f)] = (userFeatureEvaluation[(u,f)] * userFeatureEvaluationCount[(u,f)] + float(r)) / (userFeatureEvaluationCount[(u,f)] + 1)
        userFeatureEvaluationCount[(u,f)] = userFeatureEvaluationCount[(u,f)] + 1
    except Exception,e: # if the key is non-initialized, do it
        if (u,f) not in userFeatureEvaluation:
            userFeatureEvaluation[(u,f)] = 0.0
        if (u,f) not in userFeatureEvaluationCount:
            userFeatureEvaluationCount[(u,f)] = 0
        userFeatureEvaluation[(u,f)] = (userFeatureEvaluation[(u,f)] * userFeatureEvaluationCount[(u,f)] + float(r)) / (userFeatureEvaluationCount[(u,f)] + 1)
        userFeatureEvaluationCount[(u,f)] = userFeatureEvaluationCount[(u,f)] + 1

def setUserEvaluationList(u,i):
     try: # need to manage the "initialization case" in which the key does not exists
        userEvaluationList[u].append(i)
     except Exception,e: # if the key is non-initialized, do it
        userEvaluationList[u] = []
        userEvaluationList[u].append(i)

def getUserEvaluationList(user):
    try:
        return userEvaluationList[user]
    except Exception,e:
        return []

def getUserFeatureEvaluation(user,feature):
    try:
        return userFeatureEvaluation[(user,feature)]
    except Exception,e:
        return 0

def getUserFeatureEvaluationCount(user,feature):
    try:
        return userFeatureEvaluationCount[(user,feature)]
    except Exception,e:
        return 0

def numDistinctItems():
    ''' Getting the number of items that we have in our icm.csv'''
    icmRdd = sc.textFile("data/icm.csv").map(lambda line: line.split(","))
    icmFirstRow = icmRdd.first()
    icmRdd = icmRdd.filter(lambda x:x != icmFirstRow)
    numDistinctItems = icmRdd.map(lambda x: map(int,x)).sortByKey(False).keys().first()
    return numDistinctItems

def padding(u): # recicle from the old recommendetions methos
    personalizedTopN = {}
    for i,v in topN:
        personalizedTopN[i] = math.log(v)
        if not i in itemFeatureslist:
            continue
        for f in itemFeatureslist[i]:
            if not ((getUserFeatureEvaluation(u,f) == 0) or ( len(getUserEvaluationList(u))== 0 ) or (getUserFeatureEvaluationCount(u,f)==0)):
                personalizedTopN[i] = personalizedTopN[i] + getUserFeatureEvaluation(u,f) / (float(getUserFeatureEvaluationCount(u,f)) / len(getUserEvaluationList(u)))
    topNPersonalized = sorted(personalizedTopN.items(), key = lambda x:x[1], reverse = True)
    count = 0
    iterator = 0
    recommendetions = ''
    while count<5:
        if not (topNPersonalized[iterator][0] in getUserEvaluationList(u)):
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
    '''Calculating the number of distinct users that rated some items
    len(set(map(lambda x: x[0],train) + userSet))=> 15373
    last element of train=>15374'''
    return 15373

def getNumRatings():
    '''Calculating the number of ratings that of the items'''
    return len(train)

def getNumMovies():
    '''Calculating the number of items that some users rated them'''
    numMovies = trainRddLoader().map(lambda r: r[1]).distinct().count()

def loadTrain():
    global train
    train = list (csv.reader(open('data/train.csv', 'rb'), delimiter = ',')) #open csv splitting field on the comma character
    del train[0] # deletion of the string header
    train = map(lambda x: map(int, x),train) # Not so straight to read...map every (sub)element of train to int

def trainRddLoader():
    '''Creating the RDD of the train.csv'''
    trainRdd = sc.textFile("data/train.csv").map(lambda line: line.split(",")) #open csv splitting field on the comma character
    '''Getting the header of trainRdd'''
    trainFirstRow = trainRdd.first()
    '''Removing the first line of train, that is userId,iteamId,rating'''
    trainRdd = trainRdd.filter(lambda x:x != trainFirstRow)
    return trainRdd

def loadUserSet():
    '''Loader of userSet'''
    global userSet
    try:
        userSet = sc.textFile("data/test.csv").map(lambda line: line.split(",")) #open csv splitting field on the comma character
        userSetFirstRow = userSet.first()
        userSet = userSet.filter(lambda x:x != userSetFirstRow).keys().map(lambda x: int(x)).collect()
    except Exception,e:
        userSet = list (csv.reader(open('data/test.csv', 'rb'), delimiter = ','))
        del userSet[0]
        userSet = map(lambda x: x[0],map(lambda x: map(int, x),userSet))

def loadUserItemEvaluation():
    global userItemEvaluation
    '''return an hashmap structured
    (K1,K2): V
    (user,film): evaluation
    this map is obtained mapping the correct field from train set into an hashmap'''
    userItemEvaluation = dict((((x[0], x[1]), x[2])) for x in map(lambda x: map(int,x),train[1:]))

def main():
    '''
    main loop
    for all the users in userSet make the recommendetions through getRecommendetions, the output of the function
    is properly sorted only the first top5 elements are considered, eventually it's possible to get padding for the
    user which getRecommendetions is unable to fill
    Includes also percentage and temporization
    '''
    loadTrain() # Load the trainset
    loadUserSet() # Load the userSet
    loadMaps() # Load the needed data structured from the train set
    loadUserAvgRating() # Load the map of the average rating per user
    loadItemAvgRating() # Load the map of the average rating per item
    loadUserItemEvaluation() # Load the map that associate (user,item) to the rating
    loadTopN()

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
