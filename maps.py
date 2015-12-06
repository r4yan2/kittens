# modules
import csv
from collections import defaultdict
import math


def get_features_global_frequency(feature):
    if feature in feature_items_list:
        len_items = float(len(item_set))
        feature_global_frequency = len(feature_items_list[feature])
        idf = math.log(len_items/feature_global_frequency, 10)
    else:
        idf = 0

    return idf


def get_num_users():
    """

    Calculating the number of distinct users that rated some items
    len(set(map(lambda x: x[0],train) + userSet))=> 15373
    last element of train=>15374

    :return:
    """
    return 15373


def get_num_active_users():
    """

    Calculating the number of users that has rated some items

    :return:
    """
    len_active_users = len(train_user_set)

    return len_active_users


def get_num_items():
    return len(item_set)


def get_num_ratings():
    """

    Calculating the number of ratings that of the items

    :return:
    """
    return len(train)


def get_evaluation(u, i):
    """

    Getting the evaluation of a specific film for a user
    :param u:
    :param i:
    :return:
    """
    try:
        return user_item_evaluation[(u, i)]
    except Exception:
        return 0  # if the rating does not exist it returns zero


def get_features_list(i):
    try:
        return item_features_list[i]
    except Exception:
        return []  # if the item does not appears it has no features


def set_user_feature_evaluation_and_count(u, f, r):
    try:  # need to manage the "initialization case" in which the key does not exists
        user_feature_evaluation[(u, f)] = (user_feature_evaluation[(u, f)] * user_feature_evaluation_count[(u, f)] + float(
            r)) / (user_feature_evaluation_count[(u, f)] + 1)
        user_feature_evaluation_count[(u, f)] += 1
    except Exception:  # if the key is non-initialized, do it
        if (u, f) not in user_feature_evaluation:
            user_feature_evaluation[(u, f)] = 0.0
        if (u, f) not in user_feature_evaluation_count:
            user_feature_evaluation_count[(u, f)] = 0
        user_feature_evaluation[(u, f)] = (user_feature_evaluation[(u, f)] * user_feature_evaluation_count[(u, f)] + float(
            r)) / (user_feature_evaluation_count[(u, f)] + 1)
        user_feature_evaluation_count[(u, f)] += 1


def set_user_evaluation_list(u, i):
    try:  # need to manage the "initialization case" in which the key does not exists
        user_evaluation_list[u].append(i)
    except Exception:  # if the key is non-initialized, do it
        user_evaluation_list[u] = []
        user_evaluation_list[u].append(i)


def set_item_evaluators_list(i, u):
    try:  # need to manage the "initialization case" in which the key does not exists
        item_evaluators_list[i].append(u)
    except Exception:  # if the key is non-initialized, do it
        item_evaluators_list[i] = []
        item_evaluators_list[i].append(u)


def get_user_evaluation_list(user):
    try:
        return user_evaluation_list[user]
    except Exception:
        return []

def get_items_never_seen():
    return items_never_seen

def get_item_evaluators_list(item):
    try:
        return item_evaluators_list[item]
    except Exception:
        return []


def get_user_feature_evaluation(user, feature):
    try:
        return user_feature_evaluation[(user, feature)]
    except Exception:
        return 0


def get_user_feature_evaluation_count(user, feature):
    try:
        return user_feature_evaluation_count[(user, feature)]
    except Exception:
        return 0

def get_item_set():
    return item_set

def get_user_to_recommend_evaluation_count():
    return sorted(Counter(map(lambda x: len(get_user_evaluation_list(x)),user_set)).items(),key=lambda x: x[1],reverse=True)

def get_user_set():
    return user_set

def get_top_n():
    return top_n

def get_top_viewed():
    return top_viewed

def load_maps():
    
    global similarities_reader
    similarities_reader = map(lambda x: map(float, x), list(csv.reader(open('data/user_based_similarities.csv', 'rb'), delimiter=',')))  # mapping every element to int
    global train
    train = list(csv.reader(open('data/train.csv', 'rb'), delimiter=','))  # splitting csv on the comma character
    del train[0]  # deletion of the string header
    train = map(lambda x: map(int, x), train)  # Not so straight to read...map every (sub)element of train to int

    global items_never_seen
    items_never_seen = set()
    items_in_train = dict((x[1], x[2]) for x in train)
    by_feature = list(csv.reader(open('data/icm.csv', 'rb'), delimiter=','))  # open csv splitting field on the commas
    del by_feature[0]  # header remove

    global item_features_list
    item_features_list = {}
    global feature_items_list
    feature_items_list = {}

    for elem in by_feature:
        elem = map(int, elem)
        if not elem[0] in items_in_train:
            items_never_seen.add(elem[0])
        if not elem[0] in item_features_list:
            item_features_list[elem[0]] = []
        item_features_list[elem[0]].append(elem[1])

        if not elem[1] in feature_items_list:
            feature_items_list[elem[1]] = []
        feature_items_list[elem[1]].append(elem[0])

    """
    Creating some maps
    userEvaluationList: the list of items already evaluated by an user
    userEvaluationCount: not needed anymore, instead use len(userEvaluationList)
    userFeatureEvaluation: the map of the rating given to a feature calculated as average of the votes received through film containing that feature
    userFeatureEvaluationCount: count of the rating given by a user to a film containing that feature
    """
    global user_set
    user_set = list(csv.reader(open('data/test.csv', 'rb'), delimiter=','))
    del user_set[0]
    user_set = map(lambda x: x[0], map(lambda x: map(int, x), user_set))
    global item_w_features_set
    global item_set
    item_set = set(map(lambda x: int(x[0]), by_feature) + map(lambda x: int(x[1]), train))
    item_w_features_set = set(map(lambda x: int(x[0]),by_feature))
    global user_feature_evaluation  # define variable as global
    user_feature_evaluation = {}
    global user_feature_evaluation_count  # define variable as global
    user_feature_evaluation_count = {}
    global user_evaluation_list  # define variable as global
    user_evaluation_list = {}
    global item_evaluators_list  # define variable as global
    item_evaluators_list = {}
    global avg_user_rating
    avg_user_rating = {}
    count_user_rating = {}
    count_user_rating = defaultdict(lambda: 0, count_user_rating)
    global avg_item_rating
    avg_item_rating = {}
    count_item_rating = {}
    count_item_rating = defaultdict(lambda: 0, count_item_rating)
    for elem in train:
        u = elem[0]
        i = elem[1]
        r = elem[2]

        try:
            avg_user_rating[u] = (avg_user_rating[u] * count_user_rating[u] + float(r)) / (
                count_user_rating[u] + 1)  # running average
        except Exception:
            avg_user_rating[u] = 0.0
            avg_user_rating[u] = (avg_user_rating[u] * count_user_rating[u] + float(r)) / (count_user_rating[u] + 1)
        count_user_rating[u] += 1

        try:
            avg_item_rating[item] = (avg_item_rating[item] * count_item_rating[item] + float(r)) / (
                count_item_rating[item] + 1)  # running average
        except Exception:
            avg_item_rating[i] = 0.0
            avg_item_rating[i] = (avg_item_rating[i] * count_item_rating[i] + float(r)) / (
                count_item_rating[i] + 1)  # running average
        count_item_rating[i] += 1

        set_user_evaluation_list(u, i)
        set_item_evaluators_list(i, u)

        if i in item_features_list:
            for f in item_features_list[i]:
                set_user_feature_evaluation_and_count(u, f, r)
    
    global train_user_set
    train_user_set = user_evaluation_list.keys()
    
    global no_newbie_train_user_set
    no_newbie_train_user_set = filter(lambda x: len(get_user_evaluation_list(x)) >= 5, train_user_set)

    global avg_user_rating_count
    items_count = map(lambda x: len(x), user_evaluation_list.values())
    avg_user_rating_count = (reduce(lambda x, y: x + y, items_count)) / get_num_users()

    global user_item_evaluation
    '''return an hashmap structured
    (K1,K2): V
    (user,film): evaluation
    this map is obtained mapping the correct field from train set into an hashmap'''
    user_item_evaluation = dict(((x[0], x[1]), x[2]) for x in train)

    """

    Insert into an hashmap the total value for each
    film calculated by summing all the rating obtained through user
    rating divided by the sum of the votes + the
    variable shrink value obtained as logarithm
    of the number of votes divided for the number
    of users in the system.

    :return:
    """
    global top_n
    summation = 0
    counter = 0
    total = {}
    last_item = train[0][1]
    for line in train:
        item = line[1]
        rating = line[2]
        if last_item != item:
            variable_shrink = math.fabs(math.log(float(counter) / get_num_active_users()))
            total[last_item] = summation / float(counter + variable_shrink)
            counter = 0
            summation = 0
            last_item = item
        summation += rating
        counter += 1
    # Sorting in descending order the list of items
    top_n = sorted(total.items(), key=lambda x: x[1], reverse=True)
    global top_viewed
    top_viewed = sorted(item_evaluators_list.items(),key=lambda x: len(x[1]),reverse=True)

def get_train_user_set():
    return train_user_set
def populate_user_similarities(user,blacklist):
    similarities = {}
    for userX, userY, similarity in similarities_reader:
        if userX == user and userY not in blacklist:
            similarities[userY] = similarity
        elif userY == user and userX not in blacklist:
            similarities[userX] = similarity
    return similarities

def get_avg_user_rating(user):
    return avg_user_rating[user]

def get_avg_item_rating(item):
    return avg_item_rating(item)

def get_feature_items_list(feature):
    return feature_items_list(feature)

load_maps()
