from collections import defaultdict

import numpy as np

import math

from maps import get_features_list,get_train_user_set,get_user_evaluation_list,get_evaluation


def get_user_based_recommendations(user):
    """
    Return the list of the recommendations

    take an user
    the method work of the vector of the seen items of the users. For all the
    users in the userSet we make similarity with the User for which we are
    recommending by:
    * taking the list of the common items
    * changing the items with the respective evaluation
    * removing already seen items
    * removing film not well evaluated
    * making a personaCorrelation between the two users to make the classification
    * TODO fill the description

    :param user:
    :return:
    """

    items_user = get_user_evaluation_list(user)  # get the vector of the seen items
    features_items_user = {}
    threshold = xrange(6, 11)  # threshold to be applied to the possible Recommendations
    similarities = {}
    count_feature = {}
    count_feature = defaultdict(lambda: 0)
    count_average = {}
    count_average = defaultdict(lambda: 0)
    avg_common_movies = {}
    avg_common_movies = defaultdict(lambda: 0.0)
    number_common_movies = {}
    number_common_movies = defaultdict(lambda: 0)
    possible_recommendations = {}
    possible_recommendations = defaultdict(list)
    evaluations_list = {}
    evaluations_list = defaultdict(list)
    blacklist = [user]
    features_avg = {}
    features_avg = defaultdict(lambda: 0)
    shrink = {}
    predictions = {}
    ratings = {}
    
    train_user_set = get_train_user_set()
    for user_iterator in train_user_set:
        skip = True  # if no common film will be found this variable will remain fixed
        # and the remaining part of the cycle will be skipped

        items_user_iterator = get_user_evaluation_list(user_iterator)[
                              :]  # same as before, this time we need
        # to get a copy of the vector (achieved through [:]) since we are going to modify it

        ratings_user = []  # will contain the evaluations of User of the common items with userIterator
        ratings_user_iterator = []  # will contain the evaluations of user_iterator
        # of the common items with user_iterator
        for item in items_user:
            features_items_user[item] = get_features_list(item)
            if item in items_user_iterator:
                skip = False
                items_user_iterator.remove(item)

        for item in items_user_iterator:
            ratings[item] = get_evaluation(user_iterator, item)
            features = get_features_list(item)
            len_features = len(features)
            if get_evaluation(user_iterator, item) in xrange(7, 11):
                possible_recommendations[user_iterator].append(item)

        if skip or len(possible_recommendations) == 0:
            blacklist.append(user_iterator)
            continue

    similarities = populate_user_similarities(user,blacklist)
    return get_user_based_predictions(user, similarities,
                                      possible_recommendations)  # we need every element to be unique


def get_user_based_predictions(user, similarities, possible_recommendations):
    """
    This method is making the predictions for a given user

    :param user:
    :param similarities:
    :param possible_recommendations:
    :return:
    """
    avg_u = maps.avg_user_rating[user]
    user_values = []
    predictions = []

    denominator = np.sum(similarities.values())
    for userIterator in similarities.keys():
        list_numerator = {}
        list_numerator = defaultdict(list)
        for item in possible_recommendations[userIterator]:
            avg2 = maps.avg_user_rating[userIterator]
            rating = maps.get_evaluation(userIterator, item)
            user_values.append(similarities[userIterator] * (rating - avg2))
            list_numerator[item].append(user_values)
        for item in possible_recommendations[userIterator]:
            prediction = avg_u + float(np.sum(list_numerator[item])) / denominator
            predictions.append((item, prediction))
    return predictions


def pearson_user_based_correlation(u, u2, list_a, list_b, shrink):
    """

    Calculating the Pearson Correlation coefficient between two given users

    :param shrink:
    :param u:
    :param u2:
    :param list_a:
    :param list_b:
    :return:
    """
    avg_u = maps.avg_user_rating[u]
    avg_u2 = maps.avg_user_rating[u2]
    list_numerator_u = map(lambda x: x - avg_u, list_a)
    list_numerator_u2 = map(lambda x: x - avg_u2, list_b)
    numerator_pearson = np.sum([elem1 * elem2 for elem1, elem2 in zip(list_numerator_u, list_numerator_u2)])
    list_den_u = map(lambda x: x ** 2, list_numerator_u)
    list_den_u2 = map(lambda x: x ** 2, list_numerator_u2)
    denominator_pearson = math.sqrt(np.sum(list_den_u)) * math.sqrt(np.sum(list_den_u2))
    if denominator_pearson == 0:
        return 0
    pearson = numerator_pearson / (denominator_pearson + shrink)
    return pearson
