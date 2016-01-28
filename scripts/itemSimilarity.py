import math
import sys
from maps import get_item_set, get_avg_user_rating, get_item_evaluators_list, get_evaluation, get_num_users
from item_based import pearson_item_based_correlation


def get_item_similarities(itemI):

    similarities = []
    
    for itemJ in get_item_set():
        if itemI != itemJ:
            usersI = get_item_evaluators_list(itemI)[:]  # take the copy of the users that evaluated itemI
            usersJ = get_item_evaluators_list(itemJ)[:]  # take the copy of the users that evaluated itemJ
    
            ratingsItemI = []  # will contain the evaluations of User of the common items with userIterator
            ratingsItemJ = []  # will contain the evaluations of userIterator of the common items with userIterator
    
            for user in usersJ:
                if user in usersI:
                    ratingsItemI.append(get_evaluation(user,itemI) - get_avg_user_rating(user))
                    ratingsItemJ.append(get_evaluation(user,itemJ) - get_avg_user_rating(user))
            if not (len(ratingsItemI) == 0):
                shrink = math.fabs(math.log(float(len(ratingsItemI)) / get_num_users()))
                similarity = pearson_item_based_correlation(ratingsItemI, ratingsItemJ, shrink)
    
                similarities.append([itemI,itemJ,similarity])
    return sorted(similarities, key=lambda x: x[2], reverse=True)[:10]
    """
    with open('data/item_similarities.csv', 'w') as fp:
        writer = csv.writer(fp, delimiter=',')
        writer.writerows(list(similarities))
    fp.close
    """
