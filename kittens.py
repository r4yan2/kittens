import sys
import math
import csv
import time
import operator
import numpy as np
from scipy.spatial.distance import cosine
from collections import defaultdict

def cos(v1, v2):
    """

    Cosine similarity, hand implementation with numpy libraries
    :param v1:
    :param v2:
    :return:
    """

    numerator = np.dot(v1, v2)
    denominator = np.sqrt(np.dot(v1, v1)) * np.sqrt(np.dot(v2, v2))
    return  numerator/denominator

def get_user_based_recommendetions(user):
    """
    Return the list of the recommendetions

    take an user
    the method work of the vector of the seen items of the users. For all the
    users in the userSet we make similarity with the User for which we are
    recommending by:
    * taking the list of the common items
    * changind the items with the respective evaluation
    * removing already seen items
    * removing film not well evaluated
    * making a personaCorrelation between the two users to make the graduatory
    * TODO fill the description

    :param user:
    :return:
    """



    itemsUser = get_user_evaluation_list(user)  # get the vector of the seen items
    features_items = {}
    threshold = xrange(6, 11)  # threshold to be applied to the possible Recommendetions
    similarities = {}
    countFeature = {}
    countFeature = defaultdict(lambda: 0, countFeature)
    countForTheAverage = {}
    countForTheAverage = defaultdict(lambda: 0, countForTheAverage)
    avgCommonMovies = {}
    avgCommonMovies = defaultdict(lambda: 0.0, avgCommonMovies)
    numberCommonMovies = {}
    numberCommonMovies = defaultdict(lambda: 0, numberCommonMovies)
    possibleRecommendetions = {}
    possibleRecommendetions = defaultdict(list)
    evaluationsList = {}
    evaluationsList = defaultdict(list)
    blacklist = []
    featuresAvg = {}
    featuresAvg = defaultdict(lambda: 0, featuresAvg)
    shrink = {}
    predictions = {}
    ratings = {}

    for userIterator in trainUserSet:
        skip = True  # if no common film will be found this variable will remain setten and the remain part of the cicle will be skipped
        if (
            userIterator == user):  # if the user which we need to make recommendetion is the same of the one in the iterator we skip this iteration
            blacklist.append(userIterator)
            continue
        itemsUserIterator = get_user_evaluation_list(userIterator)[
                            :]  # same as before, this time we need to get a copy of the vector (achieved through [:]) since we are going to modify it
        ratingsUser = []  # will contain the evaluations of User of the common items with userIterator
        ratingsUserIterator = []  # will contain the evaluations of userIterato of the common items with userIterator
        for item in itemsUser:
            features_items[item] = get_features_list(item)
            if item in itemsUserIterator:
                skip = False
                itemsUserIterator.remove(item)

        for item in itemsUserIterator:
            ratings[item] = get_evaluation(userIterator,item)
            features = get_features_list(item)
            len_features = len(features)
            if len(features) == 0:
                continue
            for item_user in itemsUser:
                binary_features_user = map(lambda x: 1 if x in features else 0, features_items[item_user])
                sum_binary_items_user = sum(binary_features_user)

                sim = float(sum_binary_items_user)/len_features
                if sim > 0.6:
                    prediction = float(get_evaluation(user,item_user)/get_evaluation(userIterator,item)) * sim
                    predictions[item] = prediction

        for item in predictions:
            features_predictions = get_features_list(item)
            for feature in features_predictions:
                rating = get_user_feature_evaluation(user, feature)
                if (rating !=0) :
                    featuresAvg[item] = (featuresAvg[item] * countFeature[item] + float(rating)) / (countFeature[item] + 1)
                    countFeature[item] = countFeature[item] + 1
            if (featuresAvg[item] in threshold and ratings[item] in threshold):
                possibleRecommendetions[userIterator].append(item)

        if (skip or len(possibleRecommendetions) == 0):
            blacklist.append(userIterator)
            continue

    for userX, userY, similarity in similaritiesReader:
        if userX == user and userY not in blacklist:
            similarities[userY] = similarity
        elif userY == user and userX not in blacklist:
            similarities[userX] = similarity

    return get_user_based_predictions(user, similarities, possibleRecommendetions)  # we need every element to be unique

def get_item_based_recommendetions(u):
    """
    Return the list of the recommendetions

    take an user
    the method work of the vector of the seen items of the usrs. For all the
    users in the trainUserSet we make similarity with the User for which we are
    recommending by:
    * taking the list of the common items
    * changing the items with the respective evaluation
    * removing already seen items
    * removing film not well evaluated
    * making a personaCorrelation between the two users to make the graduatory
    * TODO fill the description

    :param u:
    :return:
    """

    threshold = xrange(6, 11)  # threshold to be applied to the possible Recommendetions
    similarities = {}
    countFeature = {}
    countFeature = defaultdict(lambda: 0, countFeature)
    featuresAvg = {}
    featuresAvg = defaultdict(lambda: 0, featuresAvg)
    similarities = {}
    similarities = defaultdict(list)
    predictions = []
    seen = set()
    for itemJ in filter(lambda x: get_evaluation(u, x) in threshold, get_user_evaluation_list(u)):  # direct filtering out the evaluation not in threshold
        features_j = get_features_list(itemJ)
        rating_itemJ = get_evaluation(u,itemJ)
        tf_idf = {}
        global_features_frequency = {}
        for feature in features_j:
            global_features_frequency[feature] = get_features_global_frequency(feature)


        if len(features_j) == 0:
            continue
        feature_local_frequency = float(1/len(features_j))

        for itemI in itemSet:
            if itemI == itemJ:
                continue

            features_i = get_features_list(itemI)

            usersI = get_item_evaluators_list(itemI)[:]  # take the copy of the users that evaluated itemI
            usersJ = get_item_evaluators_list(itemJ)[:]  # take the copy of the users that evaluated itemJ

            preRatingsItemI = []  # will contain the evaluations of User of the common items with userIterator
            preRatingsItemJ = []  # will contain the evaluations of userIterator of the common items with userIterator

            for user in usersJ:
                if user in usersI:
                    preRatingsItemI.append((user, itemI))
                    preRatingsItemJ.append((user, itemJ))

            '''for (user,itemI),(dontcare,itemJ) in zip(preRatingsItemI,preRatingsItemJ):
                features = getFeaturesList(itemI)
                for feature in features:
                    rating = getUserFeatureEvaluation(u,feature)
                    featuresAvg[itemI] = (featuresAvg[itemI] * countFeature[itemI] + float(rating)) / (countFeature[itemI] + 1)
                    countFeature[itemI] = countFeature[itemI] + 1
                if featuresAvg[itemI] not in threshold:
                    preRatingsItemI.remove((user,itemI))
                    preRatingsItemJ.remove((user,itemJ))'''

            if len(preRatingsItemI) == 0:
                continue
            preRatingsItemsI = filter(lambda (x,y): get_evaluation(x,y) in threshold, preRatingsItemI)
            preRatingsItemsJ = filter(lambda (x,y): get_evaluation(x,y) in threshold, preRatingsItemJ)
            ratingsItemI = map(lambda x: get_evaluation(x[0], x[1]) - avgItemRating[x[1]], preRatingsItemI)
            ratingsItemJ = map(lambda x: get_evaluation(x[0], x[1]) - avgItemRating[x[1]], preRatingsItemJ)

            binary_features_j = map(lambda x: 1 if x in features_i else 0, features_j)
            sum_binary_j = sum(binary_features_j)

            len_features_i = len(features_i)
            if len_features_i == 0:
                continue

            sim = float(sum_binary_j)/len_features_i
            for feature in global_features_frequency:
                tf_idf[feature] = feature_local_frequency * global_features_frequency[feature]
            prediction = rating_itemJ * sim
            predictions.append((itemI,prediction))
    predictions = [item for item in predictions if item[0] not in seen and not seen.add(item[0])]
    return predictions  # we need every element to be unique


def get_user_based_predictions(user, similarities, possibleRecommendetions):
    """
    This method is making the predictions for a given user

    :param user:
    :param similarities:
    :param possibleRecommendetions:
    :return:
    """
    avgu = avgUserRating[user]
    userValues = []
    predictions = []

    denominator = np.sum(similarities.values())
    for userIterator in similarities.keys():
        listNumerator = {}
        listNumerator = defaultdict(list)
        for item in possibleRecommendetions[userIterator]:
            avg2 = avgUserRating[userIterator]
            rating = get_evaluation(userIterator, item)
            userValues.append(similarities[userIterator] * (rating - avg2))
            listNumerator[item].append(userValues)
        for item in possibleRecommendetions[userIterator]:
            prediction = avgu + float(np.sum(listNumerator[item])) / denominator
            predictions.append((item, prediction))
    return predictions


def get_item_based_predictions(user, similarities):
    """
    This method is making the predictions for a given pair of items

    :param user:
    :param similarities:
    :return:
    """
    predictions = []
    for itemI in similarities.keys():
        possibleRecommendetions = similarities[itemI]
        listNumerator = []
        listDenominator = []
        for elem in possibleRecommendetions:
            itemJ = elem[0]
            similarity = elem[1]
            listNumerator.append(get_evaluation(user, itemJ) * similarity)
            listDenominator.append(similarity)
        predictions.append((itemI, float(np.sum(listNumerator)) / (np.sum(listDenominator))))
    return predictions

def get_TopN_Personalized(u, recommendetions):  # recycle from the old recommendetions methods
    personalizedTopN = {}
    user_rated_items = get_user_evaluation_list(u)

    for i, v in topN:
        personalizedTopN[i] = math.log(v,10)
        for f in get_features_list(i):
            user_feature_rating = get_user_feature_evaluation(u,f)
            number_feature_rated = get_user_feature_evaluation_count(u, f)
            number_items_seen = len(get_user_evaluation_list(u))
            if not ((user_feature_rating == 0) or ( number_items_seen == 0) or (
                 number_feature_rated == 0)):
                personalizedTopN[i] = personalizedTopN[i] + user_feature_rating * (
                    float(number_feature_rated) / number_items_seen)
    topNPersonalized = sorted(personalizedTopN.items(), key=lambda x: x[1], reverse=True)
    count = len(recommendetions)
    iterator = 0
    while count < 5:
        if not ((topNPersonalized[iterator][0] in get_user_evaluation_list(u)) or (
            topNPersonalized[iterator][0] in recommendetions)):
            recommendetions.append(topNPersonalized[iterator])
            count = count + 1
        iterator = iterator + 1
    return recommendetions


def padding_never_seen(user, recommendetions):
    count = len(recommendetions)
    iterator = 0
    possibleRecommendetions = []

    for item in itemsNeverSeen:
        features = get_features_list(item)
        featuresRatings = map(lambda x: get_user_feature_evaluation(user, x), features)
        binaryFeaturesRatings = map(lambda x: 1 if x>0 else 0, featuresRatings)

        features = filter(lambda x: x>0, ([a*b for a,b in zip(binaryFeaturesRatings,features)]))
        featuresRatings = filter(lambda x: x>0, featuresRatings)

        shrink = map(lambda x: float(get_user_feature_evaluation_count(user, x))/len(get_user_evaluation_list(user)), features)
        if len(featuresRatings) == 0:
            continue

        rating = (sum([a*b for a,b in zip(featuresRatings,shrink)]))/len(featuresRatings)

        possibleRecommendetions.append((item, rating))
    if len(possibleRecommendetions) == 0:
        return recommendetions
    possibleRecommendetions = sorted(possibleRecommendetions, key=lambda x: x[1], reverse=True)[:5]
    count = min(len(possibleRecommendetions), 5 - count)
    while count > 0:
        recommendetions.append(possibleRecommendetions[iterator])
        count -= 1
        iterator += 1
    return recommendetions


def pearson_user_based_correlation(u, u2, listA, listB, shrink):
    """

    Calculating the Pearson Correlation coefficient between two given users

    :param u:
    :param u2:
    :param listA:
    :param listB:
    :return:
    """
    avgu = avgUserRating[u]
    avgu2 = avgUserRating[u2]
    for item1, item2 in zip(listA, listB):
        listNumeratorU = map(lambda listA: listA - avgu, listA)
        listNumeratorU2 = map(lambda listB: listB - avgu2, listB)
        numeratorPearson = np.sum([elem1 * elem2 for elem1, elem2 in zip(listNumeratorU, listNumeratorU2)])
        listDenU = map(lambda listNumeratorU: listNumeratorU ** 2, listNumeratorU)
        listDenU2 = map(lambda listNumeratorU2: listNumeratorU2 ** 2, listNumeratorU2)
        denominatorPearson = math.sqrt(np.sum(listDenU)) * math.sqrt(np.sum(listDenU2))
        if denominatorPearson == 0:
            return 0
        pearson = numeratorPearson / (denominatorPearson + shrink)
    return pearson


def pearson_item_based_correlation(itemI, itemJ, listA, listB, shrink):
    """

    Calculating the Pearson Correlation coefficient between two given items

    :param itemI:
    :param itemJ:
    :param listA:
    :param listB:
    :param shrink:
    :return:
    """
    numeratorPearson = np.sum([elem1 * elem2 for elem1, elem2 in zip(listA, listB)])
    listDenI = map(lambda listA: listA ** 2, listA)
    listDenJ = map(lambda listB: listB ** 2, listB)
    denominatorPearson = math.sqrt(np.sum(listDenI)) * math.sqrt(np.sum(listDenJ))
    if denominatorPearson == 0:
        return 0
    pearson = numeratorPearson / (denominatorPearson + shrink)
    return pearson


def result_writer(result,filename):
    """

    Writing Results

    :param result:
    :return:
    """
    with open('data/'+filename, 'w') as fp:
        writer = csv.writer(fp, delimiter=',')
        writer.writerow(['userId,testItems'])
        writer.writerows(result)
    fp.close

def main(algorithm):
    """

    main loop
    for all the users in userSet make the recommendetions through getRecommendetions, the output of the function
    is properly sorted only the first top5 elements are considered, eventually it's possible to get padding for the
    user which getRecommendetions is unable to fill
    Includes also percentage and temporization

    :return:
    """
    resultToWrite = []
    loopTime = time.time()
    statsPadding = 0

    print "Making "+algorithm+" Based Recommendetions"

    for user in userSet:
        print user
        completion = float(userSet.index(user) * 100) / len(userSet)
        # sys.stdout.write("\r%f%%" % completion)
        # sys.stdout.flush()

        loopTime = time.time()
        recommendetions = []
       
        if algorithm == "user":
            countSeen = len(get_user_evaluation_list(user))
            if countSeen <= 5:
                recommendetions = get_TopN_Personalized(user, recommendetions)
            elif countSeen > 5 and countSeen <= 11:
                recommendetions = get_user_based_recommendetions(user)
                recommendetions = sorted(recommendetions, key=lambda x: x[1], reverse=True)[:5]
                if (len(recommendetions) < 5):
                    recommendetions = get_TopN_Personalized(user, recommendetions)
            elif countSeen > 11:
                recommendetions = padding_never_seen(user,recommendetions)
 
        elif algorithm == "item":
             recommendetions = get_item_based_recommendetions(user)
             recommendetions = sorted(recommendetions, key=lambda x: x[1], reverse=True)[:5]
             statsPadding = statsPadding + 5 - len(recommendetions)
             if (len(recommendetions) < 5):
                 recommendetions = padding_never_seen(user, recommendetions)
 
        else:
             print "Invalid Argument"
             return -1
 
        # writing actual recommendetion string
        recommend = ''
        for i, v in recommendetions:
            recommend = recommend + (str(i) + ' ')
        print recommend
        elem = []
        elem.append(user)
        elem.append(recommend)
        resultToWrite.append(elem)
        print "Completion percentage %f, increment %f" % (completion, time.time() - loopTime)
    result_writer(resultToWrite, algorithm+"_based_result.csv")
    print "Padding needed for %f per cent of recommendetions" % ((float(statsPadding * 100)) / (get_num_users() * 5))

disclaimer = """
    --> Kitt<3ns main script to make recommendetions <--

    To use the algorithm please make sure that maps.py load correctly (execfile("maps.py"))
    then execute main(algorithm) where algorithm is one of the following:
    'user' to make user based similarities
    'item' to make item based similarities
    """
print disclaimer
