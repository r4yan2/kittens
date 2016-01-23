from maps import get_user_evaluation_list, get_num_items, get_feature_items_list, get_item_set, get_features_list, \
    get_features_global_frequency


def get_binary_based_recommendations(user):
    recommendations = {}
    item = get_user_evaluation_list(user)[0]
    features = get_features_list(item)
    num_features = len(features)
    if num_features == 0:
        return recommendations.items()
    tf = 1.0 / num_features
    tf_idf = map(lambda feature: get_features_global_frequency(feature) * tf, features)

    for item_iterator in get_item_set():
        if item == item_iterator:
            continue
        features_item_iterator = get_features_list(item_iterator)
        binary_features = map(lambda x: 1 if x in features_item_iterator else 0, features)
        num_features_item_iterator = len(features_item_iterator)
        if num_features_item_iterator == 0:
            continue
        similarity = sum([a * b for a, b in zip(binary_features, tf_idf)]) / num_features_item_iterator
        recommendations[item_iterator] = similarity
    return recommendations.items()
