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


