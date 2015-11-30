import sys
import math
import csv
import time
import operator
import numpy as np
from scipy.spatial.distance import cosine
from collections import defaultdict

#user defined
from writer import result_writer
from user_based import get_user_based_recommendetions
from item_based import get_item_based_recommendetions
from topN import get_TopN_Personalized
from never_seen import recommend_never_seen

def cos(v1, v2):
    """

    Cosine similarity, hand implementation with numpy libraries
    :param v1:
    :param v2:
    :return:
    """

    numerator = np.dot(v1, v2)
    denominator = np.sqrt(np.dot(v1, v1)) * np.sqrt(np.dot(v2, v2))
    return  numerator/denominator

def main(*args):
    """

    main loop
    for all the users in userSet make the recommendetions through getRecommendetions, the output of the function
    is properly sorted only the first top5 elements are considered, eventually it's possible to get padding for the
    user which getRecommendetions is unable to fill
    Includes also percentage and temporization

    :return:
    """
    debug = 0
    if True in args[0]:
        debug = 1
        loopTime = time.time()
        statsPadding = 0

    resultToWrite = []
    print "Making Recommendetions"

    for user in userSet:

        completion = float(userSet.index(user) * 100) / len(userSet)

        if debug:
            print user
            loopTime = time.time()
        else:
            sys.stdout.write("\r%f%%" % completion)
            sys.stdout.flush()

        recommendetions = []
        countSeen = len(get_user_evaluation_list(user))

        if countSeen >= 0 and countSeen <3:
            recommendetions = get_TopN_Personalized(user,recommendetions)

        elif countSeen >= 3 and countSeen < 6:
            recommendetions = get_item_based_recommendetions(user)

        elif countSeen >= 6 and countSeen < 12:
            recommendetions = get_user_based_recommendetions(user)

        elif countSeen > 11:
            recommendetions = recommend_never_seen(user,recommendetions)

        recommendetions = sorted(recommendetions, key=lambda x: x[1], reverse=True)[:5]

        if (len(recommendetions) < 5):
            recommendetions = get_TopN_Personalized(user, recommendetions)
           # writing actual recommendetion string
        recommend = ''
        for i, v in recommendetions:
            recommend = recommend + (str(i) + ' ')
        if debug:
            print recommend
        elem = []
        elem.append(user)
        elem.append(recommend)
        resultToWrite.append(elem)
        print "Completion percentage %f, increment %f" % (completion, time.time() - loopTime)
    result_writer(resultToWrite, "result.csv")
    print "Padding needed for %f per cent of recommendetions" % ((float(statsPadding * 100)) / (get_num_users() * 5))

disclaimer = """
    --> Kitt<3ns main script to make recommendetions <--

    To use the algorithm please make sure that maps.py load correctly (execfile("maps.py"))
    then execute main([Boolean]) where the algorithm is choosen automagically on the user's evaluation list:
    0 <= x < 3 Top-N Personalized
    3 <= x < 6 Item-Based Content Based Filtering
    6 <= x < 12 User-Based Neighborghood
    12 <= x Never-Seen
    
    NOTE:
    - Optionally Boolean can be set to True to enable debug info
    """
print disclaimer
