import numpy
from maps import get_item_set, get_features_list, get_features_partial_frequency
import warnings


def get_tf_idf_based_recommendations(user):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category = RuntimeWarning)
        possible_recommendations = []
        for item in get_item_set():
            features = get_features_list(item)
            prediction = numpy.mean(map(lambda x: get_features_partial_frequency(user, x), features)) 
            possible_recommendations.append([item, prediction])
        sortered = sorted(possible_recommendations, key = lambda x: x[1], reverse = True)[:5]
    return map(lambda x: x[0],sortered)
