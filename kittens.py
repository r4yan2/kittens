import numpy as np
import sys
import time
import math
# user defined
from topN import *
from maps import *
from writer import result_writer
from user_based import get_user_based_recommendations
from item_based import get_item_based_recommendations
from never_seen import recommend_never_seen
from binary_based import get_binary_based_recommendations

def main(*args):
    """

    main loop
    for all the users in userSet make the recommendations through get_recommendations, the output of the function
    is properly sorted only the first top5 elements are considered, eventually it's possible to get padding for the
    user which get_recommendations is unable to fill
    Includes also percentage

    :return:
    """
    debug = args[0]
    start_time = time.time()

    if debug:
        loop_time = time.time()

    stats_padding = 0

    result_to_write = []
    print "Making recommendations"
    user_set = get_user_set()
    for user in user_set:

        completion = float(user_set.index(user) * 100) / len(user_set)
        padding = float(stats_padding * 100) / (get_num_users() * 5)

        if debug:
            loop_time = time.time()
        else:
            sys.stdout.write("\r%f%%" % completion)
            sys.stdout.flush()

        recommendations = []
        """
        count_seen = len(get_user_evaluation_list(user))

        if count_seen == 1:
            recommendations = get_binary_based_recommendations(user)

        elif 1 < count_seen < 3:
            recommendations = get_top_n_personalized(user, recommendations)

        elif 3 <= count_seen < 6:
            recommendations = get_item_based_recommendations(user)

        elif 6 <= count_seen < 12:
            try:
                recommendations = get_user_based_recommendations(user)
            except ZeroDivisionError:
                recommendations = get_item_based_recommendations(user)
        elif count_seen > 11:
            recommendations = recommend_never_seen(user, recommendations)

        if len(recommendations) < 5:
            stats_padding += 5 - len(recommendations)
            recommendations = recommend_never_seen(user, recommendations)
        """
        recommendations = get_binary_based_recommendations(user)
        #recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)
        recommend = ''
        recommended = []
        count = 0
        iterator = 0
        while (count < 5)
            (item,value) = recommendations.pop(iterator % (len(get_user_evaluation_list(user))-1))
            iterator += 1
            if item not in recommended:
                recommend += str(item) + ' '
                recommended.append(item)
                count += 1

        if debug:
            print user
            print recommend
            print "Completion percentage %f, increment %f, padding %f" % (completion, time.time() - loop_time, padding)

        elem = [user, recommend]
        result_to_write.append(elem)
    result_writer(result_to_write, "result.csv")
    print "\n100% Completed\nResult writed to file correctly\nPadding needed for %f per cent of recommendations\nCompletion time %f" % (
    padding, time.time() - start_time)


disclaimer = """
    --> Kitt<3ns main script to make recommendations <--

    To use the algorithm please make sure that maps.py load correctly (execfile("maps.py"))
    then execute main([Boolean])
    
    The algorithm execute first a pass to fill the URM with prediction obtained by CBF and then
    a second pass to extract real recommendetions using CF (Pearson)

    An Hybrid recommender system using weighted hybrid system between CBF and CF techniques

    NOTE:
    - Optionally Boolean can be set to True to enable debug info
    """
print disclaimer
