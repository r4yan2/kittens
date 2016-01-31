from maps import get_features_list, get_features_global_frequency, get_user_evaluation_list, get_item_set, get_evaluation, get_avg_user_rating
import numpy as np

def feature_correlation(user):
    items = get_user_evaluation_list(user)
    map = {}
    for item in items:
        features_list = get_features_list(item)
        for item2 in get_item_set():
            if item2 != item:
                features = get_features_list(item2)
                count = 0 
                for f in features:
                    if f in features_list:
                        count = count +1
                try:
                    if (map[item2][0] < count) and (get_evaluation(user,item) > get_evaluation(user,map[item2][1])):
                        map[item2] = [count,item]
                except KeyError:                
                    map[item2] = [count,item]

    recommendations = sorted(map.items(), key = lambda x: x[1][0], reverse = True)
    
    last = recommendations[0][1][0]
    global max_count_features
    max_count_features = recommendations[0][1][0]
    partial_count = 0
    total_count = 0
    to_push = []
    stack = []
    for recommendation in recommendations:
        if recommendation[1][0] == last :
            to_push.append(recommendation)
            partial_count += 1
        else:
            
            total_count += partial_count
            stack.extend(to_push)

            if total_count < 5:
                last = recommendation[1][0]
                to_push = [recommendation]
                partial_count = 1
            else :
                return stack


def tf_idf(x,y):
    return ((1.0 / len(get_features_list(x))) * get_features_global_frequency(y))


def get_recommendations(elem, counter,user):
    
    mean = map(lambda x: (x[0], np.mean(map(lambda y: tf_idf(x[0],y), get_features_list(x[0]))) * get_evaluation(user,x[1][1]) * (float(x[1][0])/max_count_features)), elem)
    ordered = sorted(mean, key = lambda x: x[1], reverse = True)[:counter]
    filtered = map(lambda x: x[0], ordered)
    return filtered


def get_new_kittens_recommendations(user):
    
    possible_recommendations = feature_correlation(user) 
    recommendations = (get_recommendations(possible_recommendations, 5, user))
    return recommendations
