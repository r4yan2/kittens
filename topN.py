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
        item = topNPersonalized[iterator][0] 
        if not ((item in get_user_evaluation_list(u)) or (
            item in recommendetions)):
            recommendetions.append(topNPersonalized[iterator])
            count = count + 1
        iterator = iterator + 1
    return recommendetions

def get_Top_Viewed(u, recommendetions):

    top_rated = sorted(item_evaluators_list.items(),key=lambda x: len(x[1]),reverse=True)
    count = len(recommendetions)
    iterator = 0
    while count < 5:
        item = top_rated[iterator][0]
        if not ((item in get_user_evaluation_list(u)) or (
            item in recommendetions)):
            recommendetions.append((item,0)) #magic number 0 needed for compatibility with recommendation parser in kittens
            count = count + 1
        iterator = iterator + 1
    return recommendetions
