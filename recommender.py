# modules
import csv
from collections import defaultdict
import math
import numpy
import sys
import time
import warnings
import random
# local modules
import helper

class Recommender:
    def __init__(self, db, test=False):

        self.db = db
        self.test = test

    def check_recommendations(self, user, recommendations):
        test_set = self.db.get_test_set()
        test = filter(lambda x: x[0] == user, test_set)
        ok = 0
        item_match = 0
        if not len(test) == 0:
            for line in test:
                item = line[1]
                rating = line[2]
                item_match += len(filter(lambda x: x[0] == item, recommendations))
                ok += len(filter(lambda x: x[0] == item and (math.fabs(x[1] - rating) <= 0.5), recommendations))

            return (ok * 100.0)/item_match
        else:
            raise LookupError

    def run(self, choice, debug=False, filename="result", start_from=0):

        start_time = time.time()
        to_write = []

        if debug:
            loop_time = time.time()
        if self.test:
            accuracy = {}
        stats_padding = 0
        explosions = 0
        print "Making recommendations"
        user_set = self.db.get_user_set()

        for user in user_set:
            if user < start_from:
                continue
            completion = float(user_set.index(user) * 100) / (len(user_set) - 1)
            padding = float(stats_padding * 100) / (self.db.get_num_users() * 5)

            if debug:
                loop_time = time.time()
            else:
                sys.stdout.write("\r%f%%" % completion)
                sys.stdout.flush()
                if completion >= 100.0:
                    print "\n"

            recommendations = []
            if choice == 0:
                recommendations = self.make_random_recommendations(user)
            elif choice == 1:
                recommendations = self.make_top_n_recommendations(user, False)
            elif choice == 2:
                recommendations = self.make_top_n_recommendations(user, True)
            elif choice == 3:
                recommendations = self.make_top_viewed_recommendations(user, recommendations)
            elif choice == 4:
                recommendations = self.make_top_n_personalized(user, recommendations)
            if debug:
                print "user",user,"recommendations:",recommendations
                print "Completion percentage %f, increment %f, padding %f, esplosioni schivate %i" % (
                completion, time.time() - loop_time, padding, explosions)
                print "\n=====================<3==================\n"
            if self.test:
                try:
                    accuracy[user] = self.check_recommendations(user, recommendations)
                except LookupError:
                    pass
                except ZeroDivisionError:
                    pass
            else:
                elem = [user]
                while len(elem < 6):
                    elem.append(recommendations.pop()[0])
                to_write.append(elem)
        if self.test:
            try:
                print "Running in test mode, avg accuracy: ", sum(accuracy.values())/float(len(accuracy.items())), "%"
            except ZeroDivisionError:
                print "Running in test mode\nMaybe is better something else..."
        else:
            with open('data/' + filename + '.csv', 'w', 0) as fp:
                writer = csv.writer(fp, delimiter=',', quoting=csv.QUOTE_NONE)
                writer.writerow(["userId", "testItems"])
                writer.writerows(to_write)
            print "\nCompleted!"
            if not debug:
                print "\nResult writed to /data/"+filename+".csv correctly\nPadding needed for %f per cent of recommendations\nCompletion time %f" % (
                    padding, time.time() - start_time)

    def make_random_recommendations(self, user):
        recommendations = []
        count = 0
        already_seen = self.db.get_user_evaluations(user)
        items_count = self.db.get_num_items()
        items_list = self.db.get_item_set()
        while count < 5:
            item = random.randint(0, items_count)
            if (item in items_list) and (item not in already_seen):
                recommendations.append(item)
                count += 1
        return recommendations

    def make_top_n_recommendations(self, user, shrink):

        top_n = self.db.get_top_n_w_shrink() if shrink else self.db.get_top_n()
        return top_n

    def make_top_n_personalized(self, u, recommendations):  # recycle from the old recommendations methods
        top_n = self.db.get_top_n_w_shrink() # get the top-N with shrink as base to be more accurate
        personalized_top_n = {} # will now personalize it basing on the feature set

        for i, v in top_n:
           personalized_top_n[i] = math.log(v, 10)
           for f in self.db.get_features(i):
               user_feature_rating = self.db.get_user_feature_evaluation(u, f)
               number_feature_rated = self.db.get_user_feature_evaluation_count(u, f)
               number_items_seen = len(self.db.get_user_evaluations(u))
               if not ((user_feature_rating == 0) or (number_items_seen == 0) or (
                           number_feature_rated == 0)):
                   personalized_top_n[i] += user_feature_rating * (
                       float(number_feature_rated) / number_items_seen)
        top_n_personalized = sorted(personalized_top_n.items(), key=lambda x: x[1], reverse=True)
        count = len(recommendations)
        iterator = 0
        while count < 5:
            if not top_n_personalized[iterator][0] in self.db.get_user_evaluations(u) + recommendations:
                recommendations.append(top_n_personalized[iterator][0])
                count += 1
            iterator += 1
        return map(lambda x: x[0],recommendations)

    def make_top_viewed_recommendations(self, user, recommendations):
        top_viewed = self.db.get_top_viewed()
        iterator = 0
        count = len(recommendations)
        already_seen = self.db.get_user_evaluations(user)
        while count < 5:
            item = top_viewed[iterator][0]
            if (item not in already_seen) and (
                        item not in recommendations):
                recommendations.append(item)
                count += 1
            iterator += 1
        return recommendations

    def recommend_never_seen(self, user, recommendations):
        count = len(recommendations)
        iterator = 0
        possible_recommendations = []

        for item in self.db.get_items_never_seen():
            # take list of features
            features = self.db.get_features_list(item)

            # map every feature with the corresponding rating given bu user
            features_ratings = map(lambda x: self.db.get_user_feature_evaluation(user, x), features)

            # map every rating to 1 if positive else 0
            binary_features_ratings = map(lambda x: 1 if x > 0 else 0, features_ratings)

            # filter out zeros from the element-wise multiplication of the previous defined two lists,
            # obtaining the list of the features rated positively by the user
            features = filter(lambda x: x > 0, ([a * b for a, b in zip(binary_features_ratings, features)]))

            # filter out zeros from the ratings of the features
            features_ratings = filter(lambda x: x > 0, features_ratings)

            # shrink term list created dividing the time a featured has been voted under the total value given by the user
            shrink = map(lambda x: float(self.db.get_user_feature_evaluation_count(user, x)) / len(
                self.db.get_user_evaluations(user)), features)
            if len(features_ratings) == 0:
                continue

            # Rating composition

            #rating = (sum([a*b for a,b in zip(features_ratings,shrink)]))/len(features_ratings)
            rating = sum(features_ratings) / len(features_ratings)
            possible_recommendations.append([item, rating])

        if len(possible_recommendations) == 0:
            return recommendations
        possible_recommendations = sorted(possible_recommendations, key=lambda x: x[1], reverse=True)
        count = min(len(possible_recommendations), 5 - count)
        while count > 0:
            recommendations.append(possible_recommendations[iterator])
            count -= 1
            iterator += 1
        return recommendations

    def make_custom_recommendations(self, debug, start_from, filename):
        """

        main loop
        for all the users in userSet make the recommendations through get_recommendations, the output of the function
        is properly sorted only the first top5 elements are considered, eventually it's possible to get padding for the
        user which get_recommendations is unable to fill
        Includes also percentage

        :return:
        """
        start_time = time.time()

        if debug:
            loop_time = time.time()

        stats_padding = 0
        explosions = 0
        result_to_write = []
        print "Making recommendations"
        user_set = self.db.get_user_set()

        with open('data/'+filename+'.csv', 'w', 0) as fp:
            writer = csv.writer(fp, delimiter=',')
            writer.writerow(['userId,testItems'])

            for user in user_set:
                if user < start_from:
                    continue
                completion = float(user_set.index(user) * 100) / len(user_set)
                padding = float(stats_padding * 100) / (self.db.get_num_users() * 5)

                if debug:
                    loop_time = time.time()
                else:
                    sys.stdout.write("\r%f%%" % completion)
                    sys.stdout.flush()

                recommendations = []

                count_seen = len(self.db.get_user_evaluations(user))
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
                if user not in self.db.get_casual_users():
                    try:
                        recommendations = self.get_tf_idf_based_recommendations(user)
                    except Exception:
                        explosions += 1
                        try:
                            recommendations = self.get_new_kittens_recommendations(user)
                        except Exception:
                            explosions +=1
                            recommendations = self.get_binary_based_recommendations(user)
                else:
                    try:
                        recommendations = self.get_new_kittens_recommendations(user)
                    except Exception:
                        explosions +=1
                        recommendations = self.get_binary_based_recommendations(user)
                if len(recommendations) == 0:
                        recommendations = self.get_top_n_personalized(user,recommendations)
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

    def make_binary_based_recommendations(self, user):
        recommendations = []
        similarities = []
        for item in self.db.get_user_evaluations(user):
            features = self.db.get_features(item)
            num_features = len(features)
            if num_features == 0:
                continue
            tf = 1.0 / num_features
            tf_idf = map(lambda feature: self.db.get_features_global_frequency(feature) * tf, features)
            similarities = []
            for item_iterator in self.db.get_item_set():
                if item == item_iterator:
                    continue
                features_item_iterator = self.db.get_features_list(item_iterator)
                binary_features = map(lambda x: 1 if x in features_item_iterator else 0, features)
                num_features_item_iterator = len(features_item_iterator)
                if num_features_item_iterator == 0:
                    continue
                similarities.append([item_iterator,sum([a * b for a, b in zip(binary_features, tf_idf)]) / num_features_item_iterator])
            #recommendations.append(sorted(similarities, key=lambda x: x[1], reverse=False))
        return map(lambda x: x[0], sorted(similarities,key=lambda x: x[1],reverse=True)[:5])

    def get_tf_idf_based_recommendations(self, user):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category = RuntimeWarning)
            possible_recommendations = []
            for item in self.db.get_item_set():
                features = self.db.get_features_list(item)
                prediction = numpy.mean(map(lambda x: self.db.get_features_partial_frequency(user, x), features))
                possible_recommendations.append([item, prediction])
            recommendations = sorted(possible_recommendations, key = lambda x: x[1], reverse = True)[:5]
        return map(lambda x: x[0],recommendations)

    def get_user_based_recommendations(self, user):
        """
        Return the list of the recommendations

        take an user
        the method work of the vector of the seen items of the users. For all the
        users in the userSet we make similarity with the User for which we are
        recommending by:
        * taking the list of the common items
        * changing the items with the respective evaluation
        * removing already seen items
        * removing film not well evaluated
        * making a personaCorrelation between the two users to make the classification
        * TODO fill the description

        :param user:
        :return:
        """

        items_user = self.db.get_user_evaluations(user)  # get the vector of the seen items
        features_items_user = {}
        threshold = xrange(6, 11)  # threshold to be applied to the possible Recommendations
        similarities = {}
        count_feature = defaultdict(lambda: 0, {})
        count_average = defaultdict(lambda: 0, {})
        avg_common_movies = defaultdict(lambda: 0.0, {})
        number_common_movies = defaultdict(lambda: 0, {})
        possible_recommendations = defaultdict(list, {})
        evaluations_list = defaultdict(list, {})
        blacklist = [user]
        features_avg = defaultdict(lambda: 0,{})
        shrink = {}
        predictions = {}
        ratings = {}
        global_possible_recommendations = set()

        train_user_set = self.db.get_train_user_set()
        for user_iterator in train_user_set:
            skip = True  # if no common film will be found this variable will remain fixed
            # and the remaining part of the cycle will be skipped

            items_user_iterator = self.db.get_user_evaluations(user_iterator)[
                                  :]  # same as before, this time we need
            # to get a copy of the vector (achieved through [:]) since we are going to modify it

            ratings_user = []  # will contain the evaluations of User of the common items with userIterator
            ratings_user_iterator = []  # will contain the evaluations of user_iterator
            # of the common items with user_iterator
            for item in items_user:
                features_items_user[item] = self.db.get_features(item)
                if item in items_user_iterator:
                    skip = False
                    items_user_iterator.remove(item)

            for item in items_user_iterator:
                ratings[item] = self.db.get_evaluation(user_iterator, item)
                features = self.db.get_features(item)
                len_features = len(features)
                if self.db.get_evaluation(user_iterator, item) in xrange(7, 11):
                    possible_recommendations[user_iterator].append(item)

            if skip or len(possible_recommendations) == 0:
                blacklist.append(user_iterator)
                continue
            global_possible_recommendations.update(possible_recommendations[user_iterator])
        similarities = self.db.populate_user_similarities(user, blacklist)
        return self.db.get_user_based_predictions(user, similarities,
                                                      possible_recommendations, global_possible_recommendations)  # we need every element to be unique

    def get_user_based_predictions(self, user, similarities, possible_recommendations, global_possible_recommendations):
        """
        This method is making the predictions for a given user

        :param user:
        :param similarities:
        :param possible_recommendations:
        :return:
        """
        avg_u = self.db.get_avg_user_rating(user)
        denominator = sum(similarities.values())
        list_numerator = defaultdict(list, {})

        for userIterator in similarities().keys():
            for item in possible_recommendations[userIterator]:
                avg2 = self.db.get_avg_user_rating(userIterator)
                rating = self.db.get_evaluation(userIterator, item)
                user_value = similarities[userIterator] * (rating - avg2)
                list_numerator[item].append(user_value)

        predictions = map(lambda x: (x, avg_u + float(sum(list_numerator[x])) / denominator),
                          global_possible_recommendations)
        return predictions

    def feature_correlation(self, user):
        items = self.db.get_user_evaluations(user)
        map = {}
        for item in items:
            features_list = self.db.get_features(item)
            for item2 in self.db.get_item_set():
                if item2 != item:
                    features = self.db.get_features(item2)
                    count = 0
                    for f in features:
                        if f in features_list:
                            count = count +1
                    try:
                        if (map[item2][0] < count) and (self.db.get_evaluation(user,item) > self.db.get_evaluation(user,map[item2][1])):
                            map[item2] = [count,item]
                    except KeyError:
                        map[item2] = [count,item]

        recommendations = sorted(map.items(), key = lambda x: x[1][0], reverse = True)

        last = recommendations[0][1][0]
        global max_count_features
        max_count_features = recommendations[0][1][0]
        partial_count = 0
        total_count = 0
        to_push = []
        stack = []
        for recommendation in recommendations:
            if recommendation[1][0] == last :
                to_push.append(recommendation)
                partial_count += 1
            else:

                total_count += partial_count
                stack.extend(to_push)

                if total_count < 5:
                    last = recommendation[1][0]
                    to_push = [recommendation]
                    partial_count = 1
                else :
                    return stack

    def tf_idf(self, x, y):
        return ((1.0 / len(self.db.get_features(x))) * self.db.get_features_global_frequency(y))

    def get_recommendations(self, elem, counter, user):

        mean = map(lambda x: (x[0], numpy.mean(map(lambda y: self.tf_idf(x[0],y), self.db.get_features(x[0]))) * self.db.get_evaluation(user,x[1][1]) * (float(x[1][0])/max_count_features)), elem)
        ordered = sorted(mean, key = lambda x: x[1], reverse = True)[:counter]
        filtered = map(lambda x: x[0], ordered)
        return filtered

    def get_new_kittens_recommendations(self, user):

        possible_recommendations = self.feature_correlation(user)
        recommendations = (self.get_recommendations(possible_recommendations, 5, user))
        return recommendations

    def get_item_based_recommendations(self, u):
        """
        Return the list of the recommendations

        take an user
        the method work of the vector of the seen items of the users. For all the
        users in the trainUserSet we make similarity with the User for which we are
        recommending by:
        * taking the list of the common items
        * changing the items with the respective evaluation
        * removing already seen items
        * removing film not well evaluated
        * making a personaCorrelation between the two users to make the classification
        * TODO fill the description

        :param u:
        :return:
        """

        threshold = xrange(6, 11)  # threshold to be applied to the possible recommendations
        similarities = {}
        count_feature = {}
        count_feature = defaultdict(lambda: 0, count_feature)
        features_avg = {}
        features_avg = defaultdict(lambda: 0, features_avg)
        similarities = {}
        similarities = defaultdict(list)
        predictions = []
        seen = set()
        for itemJ in filter(lambda x: self.db.get_evaluation(u, x) in threshold,
                            self.db.get_user_evaluations(u)):  # direct filtering out the evaluation not in threshold
            features_j = self.db.get_features(itemJ)
            rating_item_j = self.db.get_evaluation(u, itemJ)
            tf_idf = {}
            global_features_frequency = {}
            for feature in features_j:
                global_features_frequency[feature] = self.db.get_features_global_frequency(feature)

            if len(features_j) == 0:
                continue
            feature_local_frequency = float(1 / len(features_j))

            for itemI in self.db.get_item_set():
                if itemI == itemJ:
                    continue

                features_i = self.db.get_features(itemI)

                users_i = self.db.get_item_evaluators(itemI)[:]  # take the copy of the users that evaluated itemI
                users_j = self.db.get_item_evaluators(itemJ)[:]  # take the copy of the users that evaluated itemJ

                pre_ratings_item_i = []  # will contain the evaluations of User of the common items with userIterator
                pre_ratings_item_j = []  # will contain the evaluations of
                # userIterator of the common items with userIterator

                for user in users_j:
                    if user in users_i:
                        pre_ratings_item_i.append((user, itemI))
                        pre_ratings_item_j.append((user, itemJ))

                if len(pre_ratings_item_i) == 0:
                    continue
                pre_ratings_items_i = filter(lambda (x, y): self.db.get_evaluation(x, y) in threshold, pre_ratings_item_i)
                pre_ratings_items_j = filter(lambda (x, y): self.db.get_evaluation(x, y) in threshold, pre_ratings_item_j)
                ratings_item_i = map(lambda x: self.db.get_evaluation(x[0], x[1]) - self.db.get_avg_item_rating(x[1]), pre_ratings_item_i)
                ratings_item_j = map(lambda x: self.db.get_evaluation(x[0], x[1]) - self.db.get_avg_item_rating(x[1]), pre_ratings_item_j)

                binary_features_j = map(lambda x: 1 if x in features_i else 0, features_j)
                sum_binary_j = sum(binary_features_j)

                len_features_i = len(features_i)
                if len_features_i == 0:
                    continue

                sim = float(sum_binary_j) / len_features_i
                for feature in global_features_frequency:
                    tf_idf[feature] = feature_local_frequency * self.db.get_global_features_frequency(feature)
                prediction = rating_item_j * sim
                predictions.append([itemI, prediction])
        #predictions = [item for item in predictions if item[0] not in seen and not seen.add(item[0])]
        return predictions  # we need every element to be unique

    def get_item_based_predictions(self, user, similarities):
        """
        This method is making the predictions for a given pair of items

        :param user:
        :param similarities:
        :return:
        """
        predictions = []
        for itemI in similarities.keys():
            possible_recommendations = similarities[itemI]
            list_numerator = []
            list_denominator = []
            for elem in possible_recommendations:
                item_j = elem[0]
                similarity = elem[1]
                list_numerator.append(self.db.get_evaluation(user, item_j) * similarity)
                list_denominator.append(similarity)
                prediction = float(sum(list_numerator)) / (sum(list_denominator))
            predictions.append((itemI, prediction))
        return predictions

    def get_kittens_recommendations(self, user):
        user_items = self.db.get_user_evaluations(user)
        similarities = self.db.get_user_similarities(user)
        recommendations = []
        possible_recommendations = []
        for user_iterator in map(lambda x: x[1], similarities):
           user_iterator_items = self.db.get_user_evaluations(user_iterator)
           possible_recommendations.extend(helper.diff(user_iterator_items, user_items))
        recommendations = self.items_similarities(user_items, set(possible_recommendations))
        return sorted(recommendations, key = lambda x: x[1], reverse = True)[:5]

    def items_similarities(self, user_items, possible_recommendations):
        for item_1 in user_items:
            features = self.db.get_features(item_1)
            num_features = len(features)
            if num_features == 0 or len(user_items)==0:
                raise ValueError("stronzo trovato!")
            tf = 1.0 / num_features
            tf_idf = map(lambda feature: self.db.get_features_global_frequency(feature) * tf, features)
            similarities = []
        for item_2 in possible_recommendations:
                features_item_2 = self.db.get_features(item_2)
                binary_features = map(lambda x: 1 if x in features_item_2 else 0, features)
                num_features_item_2 = len(features_item_2)
                if num_features_item_2 == 0:
                    continue
                similarities.append([item_2,sum([a * b for a, b in zip(binary_features, tf_idf)]) / num_features_item_2])
        return similarities
