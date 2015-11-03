import math
import csv
# Creating the RDD of the train.csv
trainRdd = sc.textFile("data/train.csv").map(lambda line: line.split(","))
# Getting the header of trainRdd
trainFirstRow = trainRdd.first()

# Removing the first line of train, that is userId,itemId,rating
trainRdd=trainRdd.filter(lambda x:x !=trainFirstRow)

trainRdd=trainRdd.map(lambda x: [x[0],x[1]])

trainRdd=trainRdd.map(lambda x: map(int,x))

# List of user items seen
listUserItems = trainRdd.groupByKey().map(lambda x: (x[0], list(x[1]))).collect()

'''
#List of users and iterable items
for i in range(len(listUserItems)):
    for j in range(i,len(listUserItems)):
        cosine_similarity[(i,j)] = get_cosine(listUserItems[i][1],listUserItems[j][1])
nRdd.groupByKey().collect()
'''

# Collaborative filtering recommenders with implementing the cosine similarity

def getSimilarity(u):
similarities=[0,0,0,0,0]
v1=listUseritems[listUserItems.index(u)][1]
    for i in range(len(listUserItems)):
        v2=listUserItems[i][1]
    "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
        sumxx, sumxy, sumyy = 0, 0, 0
        vMax = max(v1,v2)
        vMin = min(v1,v2)
        for i in range(len(vMax)):
            x = vMax[i]
            if vMax[i] in vMin:
                y = vMax[i]
            else:
                y=0
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
        if (sumyy==0):
            continue
        similarty=float(sumxy)/((math.sqrt(sumxx))*(math.sqrt(sumyy)))
        for j in similarities:
            if similarity>j:
                similarities[similarities.index(j)]=similarity
                break
   
cosine_similarity = {}
for i in range(len(listUserItems)):
    for j in range(i,len(listUserItems)):
        cosine_similarity[(i,j)] = get_cosine(listUserItems[i][1],listUserItems[j][1])

userSet = sc.textFile("data/test.csv").map(lambda line: line.split(","))
userSet.filter(lambda x:x !=userSet.first())
for u in userSet.map(lambda x: map(int,x)).keys():
    getSimilarity(u)

'''
# Defining cosine similarity function

def cosine_similarity(v1,v2):
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in v1:
        x = v1[i]; y = v2[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
        return sumxy/math.sqrt(sumxx*sumyy)


# Calculating the number of ratings that has rated some items
numRatings = trainRdd.count()
# Calculating the number of distinct users that rated some items
numUsers = trainRdd.map(lambda r: r[0]).distinct().count()
# Calculating the number of items that some users rated them
numMovies = trainRdd.map(lambda r: r[1]).distinct().count()
print "Got %d ratings from %d users on %d movies." % (numRatings, numUsers, numMovies)
'''