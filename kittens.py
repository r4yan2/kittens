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

def get_user_extended_evaluation_vector(u):
    """
    Return the 20k long vector with the user's rating
    at index corresponding to related item,
    0 if no evaluation is provided

    :param u:
    :return:
    """
    evaluatedItems = getUserEvaluatedItems(u)
    userExtendedEvaluationVector = [0] * moviesNumber  # initialization
    for (user, item), value in userEvaluationList:
        if (user == u and item in evaluatedItems):
            userExtendedEvaluationVector[item - 1] = value
    return userExtendedEvaluationVector


def get_evaluation(u, i):
    """

    Getting the evaluation of a specific film for a user
    :param u:
    :param i:
    :return:
    """
    return userItemEvaluation[(u, i)]


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
    threshold = xrange(7, 11)  # threshold to be applied to the possible Recommendetions
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

    for userIterator in userSet:
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
            if item in itemsUserIterator:
                numberCommonMovies[userIterator] += 1
                skip = False
                userInverseFrequency = (math.log(get_num_users(),10)/len(get_item_evaluators_list(item)))
                ratingsUser.append(get_evaluation(user, item) * userInverseFrequency)
                ratingsUserIterator.append(get_evaluation(userIterator, item) * userInverseFrequency)
                itemsUserIterator.remove(item)

        for item in itemsUserIterator:
            features = get_features_list(item)
            for feature in features:
                rating = get_user_feature_evaluation(user, feature)
                if (rating != 0):
                    featuresAvg[item] = (featuresAvg[item] * countFeature[item] + float(rating)) / (countFeature[item] + 1)
                    countFeature[item] = countFeature[item] + 1
            if (featuresAvg[item] in threshold) and (get_evaluation(userIterator, item) in threshold) :
                possibleRecommendetions[userIterator].append(item)

        if (skip or len(possibleRecommendetions) == 0):
            blacklist.append(userIterator)
            continue

        evaluationsList[userIterator].append(ratingsUser)
        evaluationsList[userIterator].append(ratingsUserIterator)

        avgCommonMovies[userIterator] = (avgCommonMovies[userIterator] * countForTheAverage[userIterator] + len(
            ratingsUserIterator)) / (countForTheAverage[userIterator] + 1)  # Running Average
        countForTheAverage[userIterator] += 1

    for userIterator in userSet:
        if (userIterator in blacklist):
            continue
        if numberCommonMovies[userIterator] >= avgCommonMovies[userIterator]:
            similarity = pearson_user_based_correlation(user, userIterator, evaluationsList[userIterator][0],
                                                        evaluationsList[userIterator][1])
        else:
            similarity = pearson_user_based_correlation(user, userIterator, evaluationsList[userIterator][0],
                                                        evaluationsList[userIterator][1]) * (
                         numberCommonMovies[userIterator] / avgCommonMovies[userIterator])  # significance weight

        if similarity > 0.60:  # taking into consideration only positive and significant similarities
            similarities[userIterator] = similarity

    return get_user_based_predictions(user, similarities, possibleRecommendetions)  # we need every element to be unique


def get_item_based_recommendetions(u):
    """
    Return the list of the recommendetions

    take an user
    the method work of the vector of the seen items of the usrs. For all the
    users in the userSet we make similarity with the User for which we are
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

    threshold = xrange(2, 10)  # threshold to be applied to the possible Recommendetions
    similarities = {}
    countFeature = {}
    countFeature = defaultdict(lambda: 0, countFeature)
    featuresAvg = {}
    featuresAvg = defaultdict(lambda: 0, featuresAvg)
    similarities = {}
    similarities = defaultdict(list)

    for itemJ in filter(lambda x: get_evaluation(u, x) in threshold,
                        get_user_evaluation_list(u)):  # direct filtering out the evaluation not in threshold
        for itemI in itemSet:
            if itemI == itemJ:
                continue

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

            ratingsItemI = map(lambda x: get_evaluation(x[0], x[1]) - avgUserRating[user], preRatingsItemI)
            ratingsItemJ = map(lambda x: get_evaluation(x[0], x[1]) - avgUserRating[user], preRatingsItemJ)

            shrink = math.fabs(math.log(float(len(preRatingsItemI)) / get_num_users()))
            similarity = pearson_item_based_correlation(itemJ, itemI, ratingsItemJ, ratingsItemI, shrink)

            if similarity > 0.60:  # taking into consideration only positive and significant similarities
                similarities[itemI].append((itemJ, similarity))
    return get_item_based_predictions(u, similarities)  # we need every element to be unique


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

def get_features_list(i):
    try:
        return itemFeaturesList[i]
    except Exception, e:
        return [] # if the item does not appears it has no features


def set_user_feature_evaluation_and_count(u, f, r):
    try:  # need to manage the "initialization case" in which the key does not exists
        userFeatureEvaluation[(u, f)] = (userFeatureEvaluation[(u, f)] * userFeatureEvaluationCount[(u, f)] + float(
            r)) / (userFeatureEvaluationCount[(u, f)] + 1)
        userFeatureEvaluationCount[(u, f)] = userFeatureEvaluationCount[(u, f)] + 1
    except Exception, e:  # if the key is non-initialized, do it
        if (u, f) not in userFeatureEvaluation:
            userFeatureEvaluation[(u, f)] = 0.0
        if (u, f) not in userFeatureEvaluationCount:
            userFeatureEvaluationCount[(u, f)] = 0
        userFeatureEvaluation[(u, f)] = (userFeatureEvaluation[(u, f)] * userFeatureEvaluationCount[(u, f)] + float(
            r)) / (userFeatureEvaluationCount[(u, f)] + 1)
        userFeatureEvaluationCount[(u, f)] = userFeatureEvaluationCount[(u, f)] + 1


def set_user_evaluation_list(u, i):
    try:  # need to manage the "initialization case" in which the key does not exists
        userEvaluationList[u].append(i)
    except Exception, e:  # if the key is non-initialized, do it
        userEvaluationList[u] = []
        userEvaluationList[u].append(i)


def set_item_evaluators_list(i, u):
    try:  # need to manage the "initialization case" in which the key does not exists
        itemEvaluatorsList[i].append(u)
    except Exception, e:  # if the key is non-initialized, do it
        itemEvaluatorsList[i] = []
        itemEvaluatorsList[i].append(u)


def get_user_evaluation_list(user):
    try:
        return userEvaluationList[user]
    except Exception, e:
        return []


def get_item_evaluators_list(item):
    try:
        return itemEvaluatorsList[item]
    except Exception, e:
        return []


def get_user_feature_evaluation(user, feature):
    try:
        return userFeatureEvaluation[(user, feature)]
    except Exception, e:
        return 0


def get_user_feature_evaluation_count(user, feature):
    try:
        return userFeatureEvaluationCount[(user, feature)]
    except Exception, e:
        return 0

def padding(u, recommendetions):  # recycle from the old recommendetions methods
    personalizedTopN = {}
    for i, v in topN:
        personalizedTopN[i] = math.log(v)
        for f in get_features_list(i):
            if not ((get_user_feature_evaluation(u, f) == 0) or (len(get_user_evaluation_list(u)) == 0) or (
                get_user_feature_evaluation_count(u, f) == 0)):
                personalizedTopN[i] = personalizedTopN[i] + get_user_feature_evaluation(u, f) / (
                    float(get_user_feature_evaluation_count(u, f)) / len(get_user_evaluation_list(u)))
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
    threshold = xrange(8, 11)
    count = len(recommendetions)
    iterator = 0
    possibleRecommendetions = []
    for item in itemsNeverSeen:
        features = get_features_list(item)
        ratings = map(lambda x: get_user_feature_evaluation(user, x), features)
        avg = float(np.sum(ratings)) / len(features)

        if avg in threshold:
            possibleRecommendetions.append((item, avg))
    if len(possibleRecommendetions) == 0:
        return recommendetions
    possibleRecommendetions = sorted(possibleRecommendetions, key=lambda x: x[1], reverse=True)[:5]
    count = min(len(possibleRecommendetions), 5 - count)
    while count > 0:
        recommendetions.append(possibleRecommendetions[iterator])
        count -= 1
        iterator += 1
    return recommendetions


def pearson_user_based_correlation(u, u2, listA, listB):
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
        pearson = numeratorPearson / denominatorPearson
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

    for ratingI, ratingJ in zip(listA, listB):
        listNumeratorI = map(lambda listA: listA, listA)
        listNumeratorJ = map(lambda listB: listB, listB)
        numeratorPearson = np.sum([elem1 * elem2 for elem1, elem2 in zip(listNumeratorI, listNumeratorJ)])
        listDenI = map(lambda listNumeratorI: listNumeratorI ** 2, listNumeratorI)
        listDenJ = map(lambda listNumeratorJ: listNumeratorJ ** 2, listNumeratorJ)
        denominatorPearson = math.sqrt(np.sum(listDenI)) * math.sqrt(np.sum(listDenJ))
        if denominatorPearson == 0:
            return 0
        pearson = numeratorPearson / (denominatorPearson + shrink)
    return pearson


def result_writer(result):
    """

    Writing Results

    :param result:
    :return:
    """
    with open('data/result.csv', 'w') as fp:
        writer = csv.writer(fp, delimiter=',')
        writer.writerow(['userId,testItems'])
        writer.writerows(result)
    fp.close


def get_num_users():
    """

    Calculating the number of distinct users that rated some items
    len(set(map(lambda x: x[0],train) + userSet))=> 15373
    last element of train=>15374

    :return:
    """
    return 15373


def get_num_ratings():
    """

    Calculating the number of ratings that of the items

    :return:
    """
    return len(train)

def main():
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
    print "Making UserBased Recommendetions"

    for user in userSet:
        print "Completion percentage %f, increment %f" % (
        float(userSet.index(user) * 100) / len(userSet), time.time() - loopTime)
        completion = float(userSet.index(user) * 100) / len(userSet)
        # sys.stdout.write("\r%f%%" % completion)
        # sys.stdout.flush()
        loopTime = time.time()
        recommend = ''
        recommendetions = get_user_based_recommendetions(user)
        recommendetions = sorted(recommendetions, key=lambda x: x[1], reverse=True)[:5]
        print user
        statsPadding = statsPadding + 5 - len(recommendetions)
        if (len(recommendetions) < 5):
            print "padding needed"
            recommendetions = padding_never_seen(user, recommendetions)
        if (len(recommendetions) < 5):
            recommendetions = padding(user, recommendetions)
        for i, v in recommendetions:
            recommend = recommend + (str(i) + ' ')
        print recommend
        elem = []
        elem.append(user)
        elem.append(recommend)
        resultToWrite.append(elem)
    result_writer(resultToWrite)
    print "Padding needed for %f per cent of recommendetions" % ((float(statsPadding * 100)) / (get_num_users() * 5))
