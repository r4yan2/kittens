def recommend_never_seen(user, recommendetions):
    count = len(recommendetions)
    iterator = 0
    possible_recommendetions = []

    for item in items_never_seen:

        # take list of fatures
        features = get_features_list(item)
        
        # map every feature with the corresponding rating given bu user
        features_ratings = map(lambda x: get_user_feature_evaluation(user, x), features)

        # map every rating to 1 if positive else 0
        binary_features_ratings = map(lambda x: 1 if x>0 else 0, features_ratings)

        # filter out zeros from the element-wise multiplication of the previous defined two lists, obtaining the list of the features rated positively by the user
        features = filter(lambda x: x>0, ([a*b for a,b in zip(binary_features_ratings,features)]))

        # filter out zeros from the ratings of the features
        features_ratings = filter(lambda x: x>0, features_ratings)

        # shrink term list created divising the time a featured has been voted under the total value given by the user
        shrink = map(lambda x: float(get_user_feature_evaluation_count(user, x))/len(get_user_evaluation_list(user)), features)
        if len(featuresRatings) == 0:
            continue

        # Rating composition

        #rating = (sum([a*b for a,b in zip(features_ratings,shrink)]))/len(features_ratings)
        rating = sum(features_ratings)/len(features_ratings)
        possible_recommendetions.append((item, rating))
    
    if len(possible_recommendetions) == 0:
        return recommendetions
    possible_recommendetions = sorted(possible_recommendetions, key=lambda x: x[1], reverse=True)
    count = min(len(possible_recommendetions), 5 - count)
    while count > 0:
        recommendetions.append(possible_recommendetions[iterator])
        count -= 1
        iterator += 1
    return recommendetions
