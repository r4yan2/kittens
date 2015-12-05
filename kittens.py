import numpy as np
import sys
import time

# user defined
from topN import *
from maps import *
from writer import result_writer
from user_based import get_user_based_recommendations
from item_based import get_item_based_recommendations
from never_seen import recommend_never_seen
from binary_based import get_binary_based_recommendations

def cos(v1, v2):
    """

    Cosine similarity, hand implementation with numpy libraries
    :param v1:
    :param v2:
    :return:
    """

    numerator = np.dot(v1, v2)
    denominator = np.sqrt(np.dot(v1, v1)) * np.sqrt(np.dot(v2, v2))
    return numerator / denominator


def main(*args):
    """

    main loop
    for all the users in userSet make the recommendations through get_recommendations, the output of the function
    is properly sorted only the first top5 elements are considered, eventually it's possible to get padding for the
    user which get_recommendations is unable to fill
    Includes also percentage

    :return:
    """
    debug = 0
    if args[0]:
        debug = 1
        loop_time = time.time()
        stats_padding = 0

    result_to_write = []
    print "Making recommendations"
    user_set = get_user_set()
    for user in user_set:

        completion = float(user_set.index(user) * 100) / len(user_set)

        if debug:
            print user
            loop_time = time.time()
        else:
            sys.stdout.write("\r%f%%" % completion)
            sys.stdout.flush()

        recommendations = []
        count_seen = len(get_user_evaluation_list(user))

        if count_seen == 1:
            recommendations = get_binary_based_recommendations(user)

        elif 1 < count_seen < 3:
            recommendations = get_top_n_personalized(user, recommendations)

        elif 3 <= count_seen < 6:
            recommendations = get_item_based_recommendations(user)

        elif 6 <= count_seen < 12:
            recommendations = get_user_based_recommendations(user)

        elif count_seen > 11:
            recommendations = recommend_never_seen(user, recommendations)

        if recommendations < 5:
            recommendations = get_top_viewed_recommendations(user,recommendations)
 
        recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)[:5]
        
        recommend = ''
        for i, v in recommendations:
            recommend += str(i) + ' '
        if debug:
            print recommend
            print "Completion percentage %f, increment %f" % (completion, time.time() - loop_time)

        elem = [user, recommend]
        result_to_write.append(elem)
    result_writer(result_to_write, "result.csv")
    print 'Padding needed for %f per cent of recommendations' % ((float(stats_padding * 100)) / (get_num_users() * 5))


disclaimer = """
    --> Kitt<3ns main script to make recommendations <--

    To use the algorithm please make sure that maps.py load correctly (execfile("maps.py"))
    then execute main([Boolean]) where the algorithm is chosen automagically on the user's evaluation list:
    0 <= x < 3 Top-N Personalized
    3 <= x < 6 Item-Based Content Based Filtering
    6 <= x < 12 User-Based Neighborhood
    12 <= x Never-Seen
    
    NOTE:
    - Optionally Boolean can be set to True to enable debug info
    """
print disclaimer
