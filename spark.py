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

# Mapping (user,item) as key and rating as value
trainRddMappedValues=trainRdd.map(lambda x: (list((x[0],x[1])),x[2]))


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
cosineThresh=0.75
recommendetions=[]
for i in range(len(listUserItems)):
    v2=getUserVector(listUserItems[i][0])

    #compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]
        y = v2[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
  #TODO chicken and egg problem, cannot multiply for similarity because it get computed at the very end of the first for loop
        if (sumxy==0):
            if (x==0) and (y in filmThresh):
                recommendetions.add([i,y*similarity])
        continue
    similarty=float(sumxy)/((math.sqrt(sumxx))*(math.sqrt(sumyy)))
    if similarity<cosineThresh:
        break
return recommendetions

# Getting the number of items that we have in our icm.csv
icmRdd = sc.textFile("data/icm.csv").map(lambda line: line.split(","))
icmFirstRow = icmRdd.first()
icmRdd=icmRdd.filter(lambda x:x !=icmFirstRow)
numDistinctItems = icmRdd.map(lambda r: r[1]).distinct().count()

# Creating the function getUserVector that returns the 20K vector with the user's ratings for each item the user has seen

def getUserVector(u):
    listItems = listUserItems[u][1]
    itemsList = [0]*numDistinctItems
        for (user,item),v in trainRddMappedValues.collect():
        if (user!=u or item not in listItems):
            continue
        listItems[item-1]=v


cosine_similarity = {}
for i in range(len(listUserItems)):
    for j in range(i,len(listUserItems)):
        cosine_similarity[(i,j)] = get_cosine(listUserItems[i][1],listUserItems[j][1])

userSet = sc.textFile("data/test.csv").map(lambda line: line.split(","))
userSet.filter(lambda x:x !=userSet.first())
for u in userSet.map(lambda x: map(int,x)).keys().collect():
    getRecommendetions(u,userSet.map(lambda x: map(int,x)).keys().collect())

# Making the recommendetions
for elem in test:
    elem=map(int,elem)
    count=0
    iterator=0
    recommendetions=getRecommendetions(elem)
    while count<5:
        if (topNPersonalized[iterator][0][0]==elem[0]):
            if not (topNPersonalized[iterator][0][1] in uel[elem[0]]):
                recommendetions=recommendetions+(str(topNPersonalized[iterator][0][1])+' ')
                count=count+1
        iterator=iterator+1
    elem.append(recommendetions)
    result.append(elem)

# Writing Results
with open ('data/result.csv', 'w') as fp:
     writer = csv.writer(fp, delimiter=',')
     writer.writerow(["userId,testItems"])
     writer.writerows(result)
fp.close

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


# Calculating the number of ratings that of the items
numRatings = trainRdd.count()
# Calculating the number of distinct users that rated some items
numUsers = trainRdd.map(lambda r: r[0]).distinct().count()
# Calculating the number of items that some users rated them
numMovies = trainRdd.map(lambda r: r[1]).distinct().count()
print "Got %d ratings from %d users on %d movies." % (numRatings, numUsers, numMovies)
'''
