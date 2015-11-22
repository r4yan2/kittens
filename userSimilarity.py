from collections import defaultdict
import csv
import math

threshold = xrange(2, 10)  # threshold for films, 1 and 10 are removed to avoid the Global Effect
similarities = [] # list in which will be stored the similarities to be written to file

for userI in trainUserSet: #userSet is the set of all the users (users in train and test)

    completion = float(trainUserSet.index(userI) * 100) / len(trainUserSet) #output percentage visualizer
    sys.stdout.write("\r%f%%" % completion)
    sys.stdout.flush()

    countAvgCommonMovies = {}
    countAvgCommonMovies = defaultdict(lambda: 0)

    avgCommonMovies = {} # hold the average count of the common movies of userI with all the other users
    avgCommonMovies = defaultdict(lambda: 0.0)

    numberCommonMovies = {} # will hold the number of common movies found between users
    numberCommonMovies = defaultdict(lambda: 0)

    possibleRecommendetions = {}
    possibleRecommendetions = defaultdict(list)

    evaluationLists = {} # will hold the pairs of lists containing rating of the two users
    evaluationLists = defaultdict(list)

    blacklist = [] # will help to filter out user with no similairty

    shrink = {} #This hashmap will store the shrink value relative to userIterator

    itemsUserI = get_user_evaluation_list(userI)  # get the vector of the evaluated items

    for userJ in trainUserSet:
        if (userJ > userI):  # since similairty is commutative we don't need to compute similairty of userJ with userI
            itemsUserJ = get_user_evaluation_list(userJ)[
                                :]  # need to get a copy of the vector (achieved through [:]) since we are going to modify it
            ratingsUserI = []  # will contain the evaluations of User of the common items with userIterator
            ratingsUserJ = []  # will contain the evaluations of userIterato of the common items with userIterator

            for item in itemsUserI:
                if item in itemsUserJ:
                    numberCommonMovies[userJ] += 1
                    userInverseFrequency = (math.log(get_num_users(),10)/len(get_item_evaluators_list(item)))
                    ratingsUserI.append(get_evaluation(userI, item) * userInverseFrequency)
                    ratingsUserJ.append(get_evaluation(userJ, item) * userInverseFrequency)

            if not (len(ratingsUserI) == 0):

                evaluationLists[userJ].append(ratingsUserI)
                evaluationLists[userJ].append(ratingsUserJ)

                avgCommonMovies[userJ] = (avgCommonMovies[userJ] * countAvgCommonMovies[userJ] + len(
                    ratingsUserJ)) / (countAvgCommonMovies[userJ] + 1)  # Running Average
                countAvgCommonMovies[userJ] += 1

                shrink[userJ] = math.fabs(math.log(float(len(ratingsUserJ)) / get_num_items()))
            else :
                blacklist.append(userJ)

    for userJ in trainUserSet:
        if (userJ > userI) and (userJ not in blacklist):
            if numberCommonMovies[userJ] >= avgCommonMovies[userJ]:

                similarity = pearson_user_based_correlation(userI, userJ, evaluationLists[userJ][0],
                                                            evaluationLists[userJ][1], shrink[userJ])
            else:
                similarity = pearson_user_based_correlation(userI, userJ, evaluationLists[userJ][0],
                                                            evaluationLists[userJ][1],shrink[userJ]) * (
                             numberCommonMovies[userJ] / avgCommonMovies[userJ]) # significance weight

            if similarity > 0.60:  # taking into consideration only positive and significant similarities
                similarities.append([userI,userJ,similarity])

with open('data/user_based_similarities.csv', 'w') as fp:
    writer = csv.writer(fp, delimiter=',')
    writer.writerows(list(similarities))
fp.close
