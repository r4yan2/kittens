import math
import csv
# Creating the RDD of the train.csv
trainRdd = sc.textFile("data/train.csv").map(lambda line: line.split(","))
# Getting the header of trainRdd
trainFirstRow = trainRdd.first()

# Removing the first line of train, that is userId,itemId,rating
trainRdd=trainRdd.filter(lambda x:x !=trainFirstRow)

# Calculating the number of ratings that of the items
numRatings = trainRdd.count()
# Calculating the number of distinct users that rated some items
numUsers = trainRdd.map(lambda r: r[0]).distinct().count()
# Calculating the number of items that some users rated them
numMovies = trainRdd.map(lambda r: r[1]).distinct().count()
# Print stats
print "Got %d ratings from %d users on %d movies." % (numRatings, numUsers, numMovies)

trainRdd=trainRdd.map(lambda x: [x[0],x[1]])

trainRdd=trainRdd.map(lambda x: map(int,x))

# Mapping (user,item) as key and rating as value
trainRddMappedValues=trainRdd.map(lambda x: (list((x[0],x[1])),x[2]))

# Get of userSet
userSet = sc.textFile("data/test.csv").map(lambda line: line.split(","))
userSetFirst = userSet.first()
userSet = userSet.filter(lambda x:x !=userSetFirst)

# List of user items seen
listUserItems = trainRdd.groupByKey().map(lambda x: (x[0], list(x[1]))).collect()

# Collaborative filtering recommenders with implementing the cosine similarity
'''
getRecommendetions(user,userSet)
work on the user vector to make the cosine similarity
if is greater than a threshold then the evaluation
of the film is recorded in the possible recommendetions
multiplied by the similarity value for that specific user
'''

def getRecommendetions(u,userSet):

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
    numDistinctItems = icmRdd.map(lambda r: r[1]).distinct().count()
    return numDistinctItems

def getUserVector(u):
# Creating the function getUserVector that returns the 20K vector with the user's ratings for each item the user has seen
    listItems = listUserItems[u]<[1]
    itemsList = [0]*numDistinctItems()
    for (user,item),v in trainRddMappedValues.collect():
        if (user!=u or item not in listItems):
            continue
        listItems[item-1]=v
    return listItems

# Making the recommendetions
for elem in userSet.keys().collect():
    elem = map(int,elem)
    recommend=[]
    recommendetions=sorted(getRecommendetions, key=lambda x:x[1], reverse=True)[:5]
    for i,v in recommendetions:
        recommend=recommend+(str(v)+' ')
    elem.append(recommendetions)
    result.append(elem)

# Writing Results
with open ('data/result.csv', 'w') as fp:
     writer = csv.writer(fp, delimiter=',')
     writer.writerow(["userId,testItems"])
     writer.writerows(result)
fp.close
