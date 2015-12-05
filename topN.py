import math
from maps import *

top_n = get_top_n()
def get_top_n_personalized(u, recommendations):  # recycle from the old recommendations methods
    personalized_top_n = {}
    user_rated_items = get_user_evaluation_list(u)

    for i, v in top_n:
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

def get_top_viewed_recommendetions(u, recommendations):

    top_viewed = get_top_viewed()
    count = len(recommendations)
    iterator = 0
    while count < 5:
        item = top_viewed[iterator][0]
        if not ((item in get_user_evaluation_list(u)) or (
            item in recommendations)):
            recommendations.append((item,0)) #magic number 0 needed for compatibility with recommendation parser in kittens
            count = count + 1
        iterator = iterator + 1
    return recommendations
