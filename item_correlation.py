from maps import get_features_list, get_features_global_frequency, get_user_evaluation_list, get_item_set
import numpy as np

def feature_correlation(items):
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
                    if (map[item2] < count):
                        map[item2] = count
                except KeyError:                
                    map[item2]=count

    recommendations = sorted(map.items(), key = lambda x: x[1], reverse = True)
    
    last = recommendations[0][1]
    partial_count = 0
    total_count = 0
    to_push = []
    stack = []
    for recommendation in recommendations:
        if recommendation[1] == last :
            to_push.append(recommendation[0])
            partial_count += 1
        else:
            
            total_count += partial_count
            stack.append(to_push)

            if total_count < 5:
                last = recommendation[1]
                to_push = [recommendation[0]]
                partial_count = 1
            else :
                return stack

def get_tf_idf(elem, counter):

    return map(lambda x: x[0], sorted(map(lambda x: 
               (x, sum((map(lambda y: 
                   (1.0/len(get_features_list(x)))*get_features_global_frequency(y), get_features_list(x))))), elem), key = lambda x: x[1], reverse = True)[:counter])

def get_new_kittens_recommendations(user):
    
    items = get_user_evaluation_list(user)
    possible_recommendations = feature_correlation(items) 
    count = 0
    recommendations = []
    for elem in possible_recommendations:
        if len(elem) <= (5 - count):
            recommendations.extend(elem) 
            count += len(elem) 
        else:
            recommendations.extend(get_tf_idf(elem, 5 - count))
    return recommendations
