import math
import csv
import time
from scipy.spatial.distance import cosine
import numpy as np
from __future__ import print_function

def resultWriter(result):
    # Writing Results
    with open ('data/result.csv', 'w') as fp:
        writer = csv.writer(fp, delimiter=',')
        writer.writerow(["userId,testItems"])
        writer.writerows(result)
    fp.close

def trainRddLoader():

    # Creating the RDD of the train.csv
    trainRdd = sc.textFile("data/train.csv").map(lambda line: line.split(","))
    # Getting the header of trainRdd
    trainFirstRow = trainRdd.first()
    # Removing the first line of train, that is userId,iteamId,rating
    trainRdd=trainRdd.filter(lambda x:x !=trainFirstRow)
    return trainRdd

def userSetLoader():
    # Get of userSet
    userSet = sc.textFile("data/test.csv").map(lambda line: line.split(","))
    userSetFirstRow = userSet.first()
    userSet = userSet.filter(lambda x:x !=userSetFirstRow).keys().map(lambda x: int(x)).collect()
    return userSet

def stats():
    # Calculating the number of ratings that of the items
    numRatings = trainRddLoader().count()
    # Calculating the number of distinct users that rated some items
    numUsers = trainRddLoader().map(lambda r: r[0]).distinct().count()
    # Calculating the number of items that some users rated them
    numMovies = trainRddLoader().map(lambda r: r[1]).distinct().count()
    # Print stats
    print "Got %d ratings from %d users on %d movies. Total items %d" % (numRatings, numUsers, numDistinctItems())

def getUserVector(u):
# Creating the function getUserVector that returns the vector with the user's rated items
    listItems = listUserItems(u)
    itemsList = [0]*moviesNumber
    for (user,item),v in trainRddMappedValuesCollected:
        if (user!=u or item not in listItems):
            continue
        itemsList[item-1]=v
    return itemsList

def getEvaluation(u,i):
    for (user,item),v in trainRddMappedValuesCollected:
        if (user == u and item == i):
            return v


def getRecommendetions(u):
    '''
    getRecommendetions(user)
    work on the user vector to make the cosine similarity
    if is greater than a threshold then the evaluation
    of the film is recorded in the possible recommendetions
    multiplied by the similarity value for that specific user
    '''
    v1=listUserItems(u)
    filmThresh=range(6,10)
    recommendetions=[]

    for u2 in userSet:
        skip=True
        if (u2 == u):
            continue
        '''if ((i,u) in globalSimilarityUsers):
            similarity = globalSImilarityValue(globalSimilarityUsers.index( (u,i) ))
            if (similarity==-1):
                globalSimilarityUsers.append( (u,i) )
                globalSimilarityValue.append(similarity)
                continue'''
        v2=listUserItems(u2)
        listA=[]
        listB=[]
        for i in list(v1):
            if i in v2:
                skip=False

                listA.append(getEvaluation(u,i))
                listB.append(getEvaluation(u2,i))
                v2.remove(i)
            v1.remove(i)
        if (skip):
            similarity = -1
            globalSimilarityUsers.append( (u,u2) )
            globalSimilarityValue.append(similarity)
            continue

        for i in list(v2):
            if getEvaluation(u2,i) not in filmThresh:
                v2.remove(i)
        #compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)
        '''sumxx, sumxy, sumyy = 0, 0, 0
        for j in range(len(v1)):
            y = v2[j]
            x = v1[j]
            if (y == 0):
                continue
            if (x == 0):
                if (y in filmThresh):
                    film.append( (y,j+1) )
                continue
            sumxy = sumxy + x*y'''
        similarity = cosine(listA, listB)
        globalSimilarityUsers.append( (u,u2) )
        globalSimilarityValue.append(similarity)
        for i in v2:
            recommendetions.append((i,getEvaluation(u2,i) * similarity))
    return recommendetions

def cos(v1, v2):
    return np.dot(v1, v2) / (np.sqrt(np.dot(v1, v1)) * np.sqrt(np.dot(v2, v2)))

def numDistinctItems():

    # Getting the number of items that we have in our icm.csv
    icmRdd = sc.textFile("data/icm.csv").map(lambda line: line.split(","))
    icmFirstRow = icmRdd.first()
    icmRdd=icmRdd.filter(lambda x:x !=icmFirstRow)
    numDistinctItems = icmRdd.map(lambda x: map(int,x)).sortByKey(False).keys().first()
    return numDistinctItems

def listUserItems(user):
    # List of user items seen
    return itemsList[usersList.index(user)]

globalSimilarityUsers=[]
globalSimilarityValue=[]
start_time = time.time()
#Load trainRdd
trainRdd=trainRddLoader()
result=[]
# Mapping (user,item) as key and rating as value
trainRddMappedValuesCollected=trainRdd.map(lambda x: map(int,x)).map(lambda x: (list((x[0],x[1])),x[2])).collect()

itemsList=trainRdd.map(lambda x: [x[0],x[1]]).groupByKey().map(lambda x: (int(x[0]), map(int,list(x[1])))).values().collect()
usersList=trainRdd.map(lambda x: [x[0],x[1]]).groupByKey().map(lambda x: (int(x[0]), map(int,list(x[1])))).keys().collect()

moviesNumber=numDistinctItems()

print time.time()-start_time
print  "Making the recommendetions"
userSet=userSetLoader()
for user in userSet:
    print("Completion percentage"+(float(userSet.index(user)*100)/len(userSet)),end='\r')
    recommend=''
    recommendetions=sorted(getRecommendetions(user), key=lambda x:x[1], reverse=True)[:5]
    for i,v in recommendetions:
        recommend=recommend+(str(i)+' ')
    elem=[]
    elem.append(user)
    elem.append(recommend)
    result.append(elem)
resultWriter(result)
