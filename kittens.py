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
from kittens_similarity import get_kittens_recommendations
from userSimilarity import get_user_similarities
from itemSimilarity import get_item_similarities
from item_correlation import get_new_kittens_recommendations
from tf_idf_recommendations import get_tf_idf_based_recommendations
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
    start_from = args[1]
    start_time = time.time()

    if debug:
        loop_time = time.time()

    stats_padding = 0
    explosions = 0
    result_to_write = []
    print "Making recommendations"
    user_set = get_user_set()
    
    with open('data/'+args[2]+'.csv', 'w', 0) as fp:
        writer = csv.writer(fp, delimiter=',')
        writer.writerow(['userId,testItems'])

        for user in user_set:
	    if (user < start_from):
                continue
            completion = float(user_set.index(user) * 100) / len(user_set)
            padding = float(stats_padding * 100) / (get_num_users() * 5)

            if debug:
                loop_time = time.time()
            else:
                sys.stdout.write("\r%f%%" % completion)
                sys.stdout.flush()

            recommendations = []
            
            count_seen = len(get_user_evaluation_list(user))
	    """
            if count_seen < 3:
                try:
            	    print "kittens"
                    recommendations = get_kittens_recommendations(user)
                except ValueError:
            	    print "top_n(<3)"
                    recommendations = get_top_n_personalized(user, recommendations)
            
            else:
                try:
            	    print "never seen"
                    recommendations = get_binary_based_recommendations(user)
                except Exception as e:
            	    print "top_N>3"
                    recommendations = get_top_n_personalized(user, recommendations)

	    if (len(recommendations) < 5):
	        recommendations = get_binary_based_recommendations(user)
            """
            if user not in motherfuckers:
                try:
                    recommendations = get_tf_idf_based_recommendations(user)
                except Exception:
                    explosions += 1
                    try:
                        recommendations = get_new_kittens_recommendations(user)
                    except Exception:
                        explosions +=1
                        recommendations = get_binary_based_recommendations(user)
            else:
                try:
                    recommendations = get_new_kittens_recommendations(user)
                except Exception:
                    explosions +=1
                    recommendations = get_binary_based_recommendations(user)
            if len(recommendations) == 0:
                    recommendations = get_top_n_personalized(user,recommendations)
            recommend = " ".join(map(str,recommendations))
            
	    """
            recommendations = sorted(recommendations, key=lambda x: x[1], reverse=False)
            recommend = ''
            recommended = []
            count = 0
            iterator = 0
            while (count < 5):
                try:
                    (item,value) = recommendations.pop()
                except IndexError:
                    recommendations = get_top_n_personalized(user, recommendations)
                    recommend = " ".join(map(lambda x: str(x[0]),recommendations))
                    break
                iterator += 1
                if item not in recommended:
                    recommend += str(item) + ' '
                    recommended.append(item)
                    count += 1
            	print "value check:"+str(value)
            """
            if debug:
                print user
                print recommend
                print "Completion percentage %f, increment %f, padding %f, esplosioni schivate %i" % (completion, time.time() - loop_time, padding, explosions)
                print "\n=====================<3==================\n"
            elem = [user, recommend]
            writer.writerow(elem)
    print "\nCompleted!"
    if not debug:
        print "\nResult writed to file correctly\nPadding needed for %f per cent of recommendations\nCompletion time %f" % (
    padding, time.time() - start_time)

def user_similarities_to_csv():
    with open('data/user_similarities.csv', 'w') as fp:
        writer = csv.writer(fp, delimiter=',')
        user_set = get_train_user_set()
	for user in user_set:
	    completion = float(user_set.index(user) * 100) / len(user_set)
            sys.stdout.write("\r%f%%" % completion)
            sys.stdout.flush()
            writer.writerows(get_user_similarities(user))
    fp.close
 
def item_similarities_to_csv():
    with open('data/item_similarities.csv', 'w') as fp:
        writer = csv.writer(fp, delimiter=',')
        item_set = get_item_set()
        for item in item_set:
            completion = float(item_set.index(item) * 100) / len(item_set)
            sys.stdout.write("\r%f%%" % completion)
            sys.stdout.flush()
            writer.writerows(get_item_similarities(item))
    fp.close 

disclaimer = """
    --> Kitt<3ns main script to make recommendations <--

    To use the algorithm please make sure that maps.py load correctly (execfile("maps.py"))
    then execute main(Boolean,Int)
    
    The algorithm execute first a pass to fill the URM with prediction obtained by CBF and then
    a second pass to extract real recommendetions using CF (Pearson)

    An Hybrid recommender system using weighted hybrid system between CBF and CF techniques

    NOTE:
    - [0] Boolean have to be set to True to enable debug info, False will show only computation percentage
    - [1] Int have to be set to the user from which start the computation
    - [2] String have to contain the name of the csv where to save results of the computation
    """
print disclaimer
