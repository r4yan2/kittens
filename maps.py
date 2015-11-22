import csv
from collections import defaultdict
import operator
import math

def get_num_users():
    """

    Calculating the number of distinct users that rated some items
    len(set(map(lambda x: x[0],train) + userSet))=> 15373
    last element of train=>15374

    :return:
    """
    return 15373

def get_num_items():
    return len(itemSet)


def get_num_ratings():
    """

    Calculating the number of ratings that of the items

    :return:
    """
    return len(train)


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

def load_maps():
    global similaritiesReader
    similaritiesReader = map(lambda x: map(float,x),list(csv.reader(open('data/user_based_similarities.csv', 'rb'), delimiter=',')))  # open csv splitting field on the comma and mapping every element to int
    global train
    train = list(csv.reader(open('data/train.csv', 'rb'), delimiter=','))  # open csv splitting field on the comma character
    del train[0]  # deletion of the string header
    train = map(lambda x: map(int, x), train)  # Not so straight to read...map every (sub)element of train to int

    global itemsNeverSeen
    itemsNeverSeen = []
    itemsInTrain = dict(((x[1], x[2])) for x in train)
    '''Parsing the item feature list'''
    byfeature = list(
        csv.reader(open('data/icm.csv', 'rb'), delimiter=','))  # open csv splitting field on the comma character
    del byfeature[0]  # header remove
    global itemFeaturesList
    itemFeaturesList = {}
    for elem in byfeature:
        elem = map(int, elem)
        if not elem[0] in itemsInTrain:
            itemsNeverSeen.append(elem[0])
        if not elem[0] in itemFeaturesList:
            itemFeaturesList[elem[0]] = []
        itemFeaturesList[elem[0]].append(elem[1])

    """
    Creating some maps
    userEvaluationList: the list of items already evaluated by an user
    userEvaluationCount: not needed anymore, instead use len(userEvaluationList)
    userFeatureEvaluation: the map of the rating given to a feature calculated as average of the votes received through film containing that feature
    userFeatureEvaluationCount: count of the rating given by a user to a film containing that feature
    """
    global userSet
    userSet = list(csv.reader(open('data/test.csv','rb'), delimiter=','))
    del userSet[0]
    userSet = map(lambda x: x[0], map(lambda x: map(int,x), userSet))

    global itemSet
    itemSet = set(map(lambda x: int(x[0]),byfeature)+map(lambda x: int(x[1]),train))

    global userFeatureEvaluation  # define variable as global
    userFeatureEvaluation = {}
    global userFeatureEvaluationCount  # define variable as global
    userFeatureEvaluationCount = {}
    global userEvaluationList  # define variable as global
    userEvaluationList = {}
    global itemEvaluatorsList  # define variable as global
    itemEvaluatorsList = {}
    global avgUserRating
    avgUserRating = {}
    countUserRating = {}
    countUserRating = defaultdict(lambda: 0, countUserRating)
    global avgItemRating
    avgItemRating = {}
    countItemRating = {}
    countItemRating = defaultdict(lambda: 0, countItemRating)
    for elem in train:
        u = elem[0]
        i = elem[1]
        r = elem[2]

        try:
            avgUserRating[u] = (avgUserRating[u] * countUserRating[u] + float(r)) / (
            countUserRating[u] + 1)  # running average
        except Exception, e:
            avgUserRating[u] = 0.0
            avgUserRating[u] = (avgUserRating[u] * countUserRating[u] + float(r)) / (countUserRating[u] + 1)
        countUserRating[u] += 1

        try:
            avgItemRating[item] = (avgItemRating[item] * countItemRating[item] + float(r)) / (
            countItemRating[item] + 1)  # running average
        except Exception, e:
            avgItemRating[i] = 0.0
            avgItemRating[i] = (avgItemRating[i] * countItemRating[i] + float(r)) / (
            countItemRating[i] + 1)  # running average
        countItemRating[i] += 1

        set_user_evaluation_list(u, i)
        set_item_evaluators_list(i, u)

        if i in itemFeaturesList:
            for f in itemFeaturesList[i]:
                set_user_feature_evaluation_and_count(u, f, r)
    global trainUserSet
    trainUserSet = userEvaluationList.keys()

    global avgUserRatingCount
    itemsCount = map(lambda x: len(x), userEvaluationList.values())
    avgUserRatingCount = (reduce(lambda x, y: x + y, itemsCount)) / get_num_users()

    global userItemEvaluation
    '''return an hashmap structured
    (K1,K2): V
    (user,film): evaluation
    this map is obtained mapping the correct field from train set into an hashmap'''
    userItemEvaluation = dict((((x[0], x[1]), x[2])) for x in train)

    """

    Insert into an hashmap the total value for each
    film calculated by summing all the rating obtained throught user
    rating divided by the sum of the votes + the
    variable shrink value obtained as logarithm
    of the number of votes divided for the number
    of users in the system.

    :return:
    """
    global topN
    sum = 0
    counter = 0
    total = {}
    lastItem = train[0][1]
    for line in train:
        item=line[1]
        rating=line[2]
        if (lastItem != item):
            variableShrink = math.fabs(math.log(float(counter) / get_num_users()))
            total[lastItem] = sum / float(counter + variableShrink);
            counter = 0
            sum = 0
            lastItem = item
        sum = sum + rating
        counter = counter + 1
    # Sorting in descending order the list of items
    topN = sorted(total.items(), key=lambda x: x[1], reverse=True)

load_maps()
