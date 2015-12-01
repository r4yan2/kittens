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

        for itemI in item_set:
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


