import math
import csv

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

def stats(trainRdd):
    # Calculating the number of ratings that of the items
    numRatings = trainRdd.count()
    # Calculating the number of distinct users that rated some items
    numUsers = trainRdd.map(lambda r: r[0]).distinct().count()
    # Calculating the number of items that some users rated them
    numMovies = trainRdd.map(lambda r: r[1]).distinct().count()
    # Print stats
    print "Got %d ratings from %d users on %d movies. Total items %d" % (numRatings, numUsers, numMovies, numDistinctItems())

def getUserVector(u):
# Creating the function getUserVector that returns the 20K vector with the user's ratings for each item the user has seen
    listItems = listUserItems[u-1][1]
    itemsList = [0]*numDistinctItems()
    for (user,item),v in trainRddMappedValuesCollected:
        if (user!=u or item not in listItems):
            continue
        itemsList[item-1]=v
    return itemsList

def getRecommendetions(u):
    '''
    getRecommendetions(user)
    work on the user vector to make the cosine similarity
    if is greater than a threshold then the evaluation
    of the film is recorded in the possible recommendetions
    multiplied by the similarity value for that specific user
    '''
    v1=getUserVector(u)
    filmThresh=range(6,10)
    recommendetions=[]
    for i in range(len(listUserItems)):
        film=[]
        v2=getUserVector(listUserItems[i][0])

    #compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
        sumxx, sumxy, sumyy = 0, 0, 0
        for i in range(len(v1)):
            x = v1[i]
            y = v2[i]
            sumxx += x*x
            sumyy += y*y
            sumxy += x*y
            if (x==0) and (y in filmThresh):
                recommendetions.append((y,i+1))
        if (sumxy==0):
            continue
        similarity=float(sumxy)/((math.sqrt(sumxx))*(math.sqrt(sumyy)))
        for rating,item in film:
            recommendetions.append((item,rating*similarity))
    return recommendetions

def numDistinctItems():

    # Getting the number of items that we have in our icm.csv
    icmRdd = sc.textFile("data/icm.csv").map(lambda line: line.split(","))
    icmFirstRow = icmRdd.first()
    icmRdd=icmRdd.filter(lambda x:x !=icmFirstRow)
    numDistinctItems = icmRdd.map(lambda r: r[0]).distinct().count()
    return numDistinctItems

#Load trainRdd
trainRdd=trainRddLoader()

# List of user items seen
listUserItems = trainRdd.map(lambda x: [x[0],x[1]]).groupByKey().map(lambda x: (int(x[0]), map(int,list(x[1])))).collect()

# Mapping (user,item) as key and rating as value
trainRddMappedValuesCollected=trainRdd.map(lambda x: map(int,x)).map(lambda x: (list((x[0],x[1])),x[2])).collect()

# Making the recommendetions
userSet=userSetLoader()
for user in userSet:
    recommend=[]
    recommendetions=sorted(getRecommendetions(user), key=lambda x:x[1], reverse=True)[:5]
    for i,v in recommendetions:
        recommend=recommend+(str(v)+' ')
    user.append(recommendetions)
    result.append(user)
resultWriter(result)
