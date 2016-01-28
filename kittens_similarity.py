from userSimilarity import get_user_similarities
from maps import get_user_evaluation_list, get_features_list, get_features_global_frequency, populate_user_similarities

def diff(a, b):
    b = set(b)
    return [aa for aa in a if aa not in b]

def get_kittens_recommendations(user):
    user_items = get_user_evaluation_list(user)
    similarities = get_user_similarities(user)
    recommendations = []
    possible_recommendations = []
    for user_iterator in map(lambda x: x[1], similarities):
       user_iterator_items = get_user_evaluation_list(user_iterator)
       possible_recommendations.extend(diff(user_iterator_items, user_items))
    recommendations = items_similarities(user_items, set(possible_recommendations))
    return sorted(recommendations, key = lambda x: x[1], reverse = True)[:5]

def items_similarities(user_items, possible_recommendations):
    for item_1 in user_items:
        features = get_features_list(item_1)
        num_features = len(features)
        if num_features == 0 or len(user_items)==0:
            raise ValueError("stronzo trovato!")
        tf = 1.0 / num_features
        tf_idf = map(lambda feature: get_features_global_frequency(feature) * tf, features)
        similarities = []
	for item_2 in possible_recommendations:
            features_item_2 = get_features_list(item_2)
            binary_features = map(lambda x: 1 if x in features_item_2 else 0, features)
            num_features_item_2 = len(features_item_2)
            if num_features_item_2 == 0:
                continue
            similarities.append([item_2,sum([a * b for a, b in zip(binary_features, tf_idf)]) / num_features_item_2])
    return similarities
