import math

import maps
from maps import get_user_evaluation_list, get_features_list, get_user_feature_evaluation, \
    get_user_feature_evaluation_count


def get_top_n_personalized(u, recommendations):  # recycle from the old recommendations methods
    personalized_top_n = {}
    user_rated_items = get_user_evaluation_list(u)

    for i, v in maps.top_n:
        personalized_top_n[i] = math.log(v, 10)
        if len(get_user_evaluation_list(u)) ==2:
            for f in get_features_list(i):
                user_feature_rating = get_user_feature_evaluation(u, f)
                number_feature_rated = get_user_feature_evaluation_count(u, f)
                number_items_seen = len(get_user_evaluation_list(u))
                if not ((user_feature_rating == 0) or (number_items_seen == 0) or (
                            number_feature_rated == 0)):
                    personalized_top_n[i] += user_feature_rating * (
                        float(number_feature_rated) / number_items_seen)
    top_n_personalized = sorted(personalized_top_n.items(), key=lambda x: x[1], reverse=True)
    count = len(recommendations)
    iterator = 0
    while count < 5:
        if not ((top_n_personalized[iterator][0] in get_user_evaluation_list(u)) or (
                    top_n_personalized[iterator][0] in recommendations)):
            recommendations.append(top_n_personalized[iterator])
            count += 1
        iterator += 1
    return recommendations
