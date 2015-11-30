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



    items_user = get_user_evaluation_list(user)  # get the vector of the seen items
    features_items_user = {}
    threshold = xrange(6, 11)  # threshold to be applied to the possible Recommendetions
    similarities = {}
    count_feature = {}
    count_feature = defaultdict(lambda: 0)
    count_average = {}
    count_average = defaultdict(lambda: 0)
    avgCommonMovies = {}
    avgCommonMovies = defaultdict(lambda: 0.0)
    numberCommonMovies = {}
    numberCommonMovies = defaultdict(lambda: 0)
    possibleRecommendetions = {}
    possibleRecommendetions = defaultdict(list)
    evaluationsList = {}
    evaluationsList = defaultdict(list)
    blacklist = [user]
    featuresAvg = {}
    featuresAvg = defaultdict(lambda: 0)
    shrink = {}
    predictions = {}
    ratings = {}

    for user_iterator in train_user_set:
        skip = True  # if no common film will be found this variable will remain setten and the remain part of the cicle will be skipped

        items_user_iterator = get_user_evaluation_list(user_iterator)[
                            :]  # same as before, this time we need to get a copy of the vector (achieved through [:]) since we are going to modify it
        ratingsUser = []  # will contain the evaluations of User of the common items with userIterator
        ratingsUserIterator = []  # will contain the evaluations of userIterato of the common items with userIterator
        for item in itemsUser:
            features_items_user[item] = get_features_list(item)
            if item in itemsUserIterator:
                skip = False
                itemsUserIterator.remove(item)

        for item in itemsUserIterator:
            ratings[item] = get_evaluation(userIterator,item)
            features = get_features_list(item)
            len_features = len(features)
            if get_evaluation(user_iterator,item) in xrange(7,11):
                possible_recommendetions[user_iterator].append(item)

        if (skip or len(possibleRecommendetions) == 0):
            blacklist.append(userIterator)
            continue

    for userX, userY, similarity in similaritiesReader:
        if userX == user and userY not in blacklist:
            similarities[userY] = similarity
        elif userY == user and userX not in blacklist:
            similarities[userX] = similarity

    return get_user_based_predictions(user, similarities, possibleRecommendetions)  # we need every element to be unique


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


