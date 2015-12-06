from collections import defaultdict

import math

from maps import get_item_evaluators_list, get_evaluation, get_user_evaluation_list, get_features_list, get_features_global_frequency

def get_item_based_recommendations(u):
    """
    Return the list of the recommendations

    take an user
    the method work of the vector of the seen items of the users. For all the
    users in the trainUserSet we make similarity with the User for which we are
    recommending by:
    * taking the list of the common items
    * changing the items with the respective evaluation
    * removing already seen items
    * removing film not well evaluated
    * making a personaCorrelation between the two users to make the classification
    * TODO fill the description

    :param u:
    :return:
    """

    threshold = xrange(6, 11)  # threshold to be applied to the possible recommendations
    similarities = {}
    count_feature = {}
    count_feature = defaultdict(lambda: 0, count_feature)
    features_avg = {}
    features_avg = defaultdict(lambda: 0, features_avg)
    similarities = {}
    similarities = defaultdict(list)
    predictions = []
    seen = set()
    for itemJ in filter(lambda x: get_evaluation(u, x) in threshold, get_user_evaluation_list(u)):  # direct filtering out the evaluation not in threshold
        features_j = get_features_list(itemJ)
        rating_item_j = get_evaluation(u, itemJ)
        tf_idf = {}
        global_features_frequency = {}
        for feature in features_j:
                global_features_frequency[feature] = get_features_global_frequency(feature)

        if len(features_j) == 0:
            continue
        feature_local_frequency = float(1/len(features_j))

        for itemI in get_item_set():
            if itemI == itemJ:
                continue

            features_i = get_features_list(itemI)

            users_i = get_item_evaluators_list(itemI)[:]  # take the copy of the users that evaluated itemI
            users_j = get_item_evaluators_list(itemJ)[:]  # take the copy of the users that evaluated itemJ

            pre_ratings_item_i = []  # will contain the evaluations of User of the common items with userIterator
            pre_ratings_item_j = []  # will contain the evaluations of
            # userIterator of the common items with userIterator

            for user in users_j:
                if user in users_i:
                    pre_ratings_item_i.append((user, itemI))
                    pre_ratings_item_j.append((user, itemJ))

            if len(pre_ratings_item_i) == 0:
                continue
            pre_ratings_items_i = filter(lambda (x, y): get_evaluation(x, y) in threshold, pre_ratings_item_i)
            pre_ratings_items_j = filter(lambda (x, y): get_evaluation(x, y) in threshold, pre_ratings_item_j)
            ratings_item_i = map(lambda x: get_evaluation(x[0], x[1]) - get_avg_item_rating(x[1]), pre_ratings_item_i)
            ratings_item_j = map(lambda x: get_evaluation(x[0], x[1]) - get_avg_item_rating(x[1]), pre_ratings_item_j)

            binary_features_j = map(lambda x: 1 if x in features_i else 0, features_j)
            sum_binary_j = sum(binary_features_j)

            len_features_i = len(features_i)
            if len_features_i == 0:
                continue

            sim = float(sum_binary_j)/len_features_i
            for feature in global_features_frequency:
                tf_idf[feature] = feature_local_frequency * global_features_frequency[feature]
            prediction = rating_item_j * sim
            predictions.append((itemI, prediction))
    predictions = [item for item in predictions if item[0] not in seen and not seen.add(item[0])]
    return predictions  # we need every element to be unique


def get_item_based_predictions(user, similarities):
    """
    This method is making the predictions for a given pair of items

    :param user:
    :param similarities:
    :return:
    """
    predictions = []
    for itemI in similarities.keys():
        possible_recommendations = similarities[itemI]
        list_numerator = []
        list_denominator = []
        for elem in possible_recommendations:
            item_j = elem[0]
            similarity = elem[1]
            list_numerator.append(get_evaluation(user, item_j) * similarity)
            list_denominator.append(similarity)
        predictions.append((itemI, float(sum(list_numerator)) / (sum(list_denominator))))
    return predictions


def pearson_item_based_correlation(list_a, list_b, shrink):
    """

    Calculating the Pearson Correlation coefficient between two given items

    :param list_a:
    :param list_b:
    :param shrink:
    :return:
    """
    numerator_pearson = sum([elem1 * elem2 for elem1, elem2 in zip(list_a, list_b)])
    list_den_i = map(lambda x: x ** 2, list_a)
    list_den_j = map(lambda x: x ** 2, list_b)
    denominator_pearson = math.sqrt(sum(list_den_i)) * math.sqrt(sum(list_den_j))
    if denominator_pearson == 0:
        return 0
    pearson = numerator_pearson / (denominator_pearson + shrink)
    return pearson


