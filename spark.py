# Creating the RDD of the train.csv
trainRdd = sc.textFile("data/train.csv").map(lambda line: line.split(","))
# Getting the header of trainRdd
trainFirstRow = trainRdd.first()

# Removing the first line of train, that is userId,itemId,rating
trainRdd = trainRdd.filter(lambda x:x !=trainFirstRow)

# Calculating the number of ratings that has rated some items
numRatings = trainRdd.count()
# Calculating the number of distinct users that rated some items
numUsers = trainRdd.map(lambda r: r[0]).distinct().count()
# Calculating the number of items that some users rated them
numMovies = trainRdd.map(lambda r: r[1]).distinct().count()
print "Got %d ratings from %d users on %d movies." % (numRatings, numUsers, numMovies)

# Creating the RDD of the uel.csv
#uelRdd = sc.textFile("data/uel.csv").map(lambda line: line.split(","))
# uelRdd.map(lambda x: map(int,x))
