#from pyspark import SparkContext
#Creating the RDD of the train.csv
trainRdd=sc.textFile("train.csv").map(lambda line: line.split(","))

#Getting the header of trainRdd
trainFirstRow = trainRdd.first()

#Removing the first line of train, that is userId,itemId,rating
trainRdd = trainRdd.filter(lambda x:x !=trainFirstRow)

#Calculating the number of users that has rated some items
numberUsers = trainRdd.count()
