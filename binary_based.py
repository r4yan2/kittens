from maps import *

recommendations = {}

def get_binary_based_recommendations(user):
    item = get_user_evaluation_list(user)[0]
    features = get_features_list(item)
    idf = map(lambda feature: math.log(float(get_num_items())/len(feature_items_list[feature]),10),features)
    
    for item_iterator in get_item_set():
        features_item_iterator = get_features_list(item_iterator)
        binary_features = map(lambda x: 1 if x in features_item_iterator else 0,features)
        num_features_item_iterator = len(features_item_iterator)
        if num_features_item_iterator == 0:
            continue
        similarity = sum([a*b for a,b in zip(binary_features,idf)])/num_features_item_iterator
        recommendations[item_iterator] = similarity
    return recommendations.items()

