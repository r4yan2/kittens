from collections import defaultdict
import csv
import math
from maps import get_train_user_set, get_avg_user_rating, get_num_items
from user_based import pearson_user_based_correlation 

threshold = xrange(2, 10)  # threshold for films, 1 and 10 are removed to avoid the Global Effect
similarities = [] # list in which will be stored the similarities found


def get_user_similarities(users):
    for userI in users: #user_set is the set of users for which we need similarities
    
        completion = float(users.index(userI) * 100) / len(users) #output percentage visualizer
        sys.stdout.write("\r%f%%" % completion)
        sys.stdout.flush()
    
        countAvgCommonMovies = {} # Needed for the running Avg
        countAvgCommonMovies = defaultdict(lambda: 0)
    
        avgCommonMovies = {} # hold the average count of the common movies of userI with all the other users
        avgCommonMovies = defaultdict(lambda: 0.0)
    
        numberCommonMovies = {} # will hold the number of common movies found between users
        numberCommonMovies = defaultdict(lambda: 0)
    
        possibleRecommendetions = {} # hold the possible recommendations (capitan ovvio colpisce ancora)
        possibleRecommendetions = defaultdict(list)
    
        evaluationLists = {} # will hold the pairs of lists containing rating of the two users
        evaluationLists = defaultdict(list)
    
        blacklist = [] # will help to filter out user with no similairty
    
        shrink = {} # This hashmap will store the shrink value relative to userIterator
    
        itemsUserI = get_user_evaluation_list(userI)  # get the vector of the evaluated items
    
        for userJ in get_train_user_set():
            if (userJ > userI):  # since similairty is commutative we don't need to compute similairty of userJ with userI
                itemsUserJ = get_user_evaluation_list(userJ)[
                                    :]  # need to get a copy of the vector (achieved through [:]) since we are going to modify it
                ratingsUserI = []  # will contain the evaluations of User of the common items with userIterator
                ratingsUserJ = []  # will contain the evaluations of userIterato of the common items with userIterator
    
                for item in itemsUserI:
                    if item in itemsUserJ:
                        numberCommonMovies[userJ] += 1
                        userInverseFrequency = (math.log(get_num_users(),10)) - math.log(len(get_item_evaluators_list(item)),10)
                        ratingsUserI.append((get_evaluation(userI, item) - get_avg_user_rating(userI)) * userInverseFrequency)
                        ratingsUserJ.append((get_evaluation(userJ, item) - get_avg_user_rating(userJ)) * userInverseFrequency)
    
                if not (len(ratingsUserI) == 0):
    
                    evaluationLists[userJ].append(ratingsUserI)
                    evaluationLists[userJ].append(ratingsUserJ)
    
                    avgCommonMovies[userJ] = (avgCommonMovies[userJ] * countAvgCommonMovies[userJ] + len(
                        ratingsUserJ)) / (countAvgCommonMovies[userJ] + 1)  # Running Average
                    countAvgCommonMovies[userJ] += 1
    
                    shrink[userJ] = math.fabs(math.log(float(len(ratingsUserJ)) / get_num_items()))
                else :
                    blacklist.append(userJ)
    
        for userJ in get_train_user_set():
            if (userJ > userI) and (userJ not in blacklist):
                if numberCommonMovies[userJ] >= avgCommonMovies[userJ]:
    
                    similarity = pearson_user_based_correlation(userI, userJ, evaluationLists[userJ][0],
                                                                evaluationLists[userJ][1], shrink[userJ])
                else:
                    similarity = pearson_user_based_correlation(userI, userJ, evaluationLists[userJ][0],
                                                                evaluationLists[userJ][1],shrink[userJ]) * (
                                 numberCommonMovies[userJ] / 50) # significance weight
    		if similarity > 0:
                    similarities.append([userJ,similarity])
	return sorted(similarities, key=lambda x: x[1], reverse=True)[:10] #10 is choosen based on the number which the coder has dreamed last night
