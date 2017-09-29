import csv
import math
from collections import defaultdict, Counter
import random

class Database:
    def __init__(self, test=False):
        self.train_csv = csv.reader(open('data/train.csv', 'rb'), delimiter=',')  # splitting csv on the comma character
        if test:
            self.test_set = []
            self.compute_test_set()
        self.user_similarities_csv = None#csv.reader(open('data/user_similarities.csv', 'rb'), delimiter=',')
        self.item_features_csv = csv.reader(open('data/icm.csv', 'rb'), delimiter=',')
        self.test_csv = csv.reader(open('data/test.csv', 'rb'), delimiter=',')
        self.generate_lists_from_icm()
        self.generate_lists_from_train()

    def get_test_set(self):
        try:
            return self.test_set
        except AttributeError:
            self.compute_test_set()
            return self.test_set

    def compute_test_set(self):
        train = self.get_train_list()
        for i in xrange(0, 20980):
            line = random.randint(0, len(train) - 1)
            self.test_set.append(train.pop(line))

    def get_train_list(self):
        try:
            return self.train_list
        except AttributeError:
            train_list = list(self.train_csv)
            del train_list[0]
            self.train_list = map(lambda x: map(int, x), train_list)
            return self.train_list

    def get_test_list(self):
        try:
            return self.test_list
        except AttributeError:
            test_list = list(self.test_csv)
            del test_list[0]
            self.test_list = map(lambda x: int(x[0]), test_list)
            return self.test_list

    def get_users_similarities_list(self):
        try:
            return self.user_similarities_list
        except AttributeError:
            user_similarities_list = list(self.user_similarities_csv)
            self.user_similarities_list = map(lambda x: map(float, x), user_similarities_list)
            return self.user_similarities_list

    def get_item_features_list(self):
        try:
            return self.item_features_list_int
        except AttributeError:
            item_features_list = list(self.item_features_csv)
            del item_features_list[0]
            self.item_features_list_int = map(lambda x: map(int,x), item_features_list)
            return self.item_features_list_int

    def get_items_in_train(self):
        try:
            return self.items_in_train
        except AttributeError:
            train_list = self.get_train_list()
            self.items_in_train = dict((x[1], x[2]) for x in train_list)
            return self.items_in_train

    def get_train_user_set(self):
        try:
            return self.train_user_set
        except AttributeError:
            train_list = self.get_train_list()
            self.train_user_set = self.get_user_evaluations_list().keys()
            return self.train_user_set

    def get_user_set(self):
        try:
            return self.user_set
        except AttributeError:
            test_list = self.get_test_list()
            self.user_set = test_list
            return self.user_set

    def get_item_set(self):
        try:
            return self.item_set
        except AttributeError:
            train_list = self.get_train_list()
            item_features_list = self.get_item_features_list()
            self.item_set = set(map(lambda x: x[0], item_features_list) + map(lambda x: x[1], train_list))
            return self.item_set

    def get_item_w_features_set(self):
        try:
            return self.item_w_features_set
        except AttributeError:
            item_features_list = self.get_item_features_list()
            self.item_w_features_set = set(map(lambda x: int(x[0]), item_features_list))
            return self.item_w_features_set

    def get_active_users_set(self):
        """an active user as at least 5 given votes"""
        try:
            return self.active_users_set
        except AttributeError:
            train_users_set = self.get_train_user_set()
            self.active_users_set = filter(lambda x: len(self.get_user_evaluations(x)) >= 5, train_users_set)
            return self.active_users_set

    def get_avg_user_rating_count(self):
        try:
            return self.avg_user_rating_count
        except AttributeError:
            user_evaluation_list = self.get_user_evaluations_list()
            items_count = map(lambda x: len(x), user_evaluation_list.values())
            self.avg_user_rating_count = (reduce(lambda x, y: x + y, items_count)) / self.get_num_users()

    def get_user_item_evaluation(self, u, i):
        '''return an hashmap structured
        (K1,K2): V
        (user,film): evaluation
        this map is obtained mapping the correct field from train set into an hashmap'''
        try:
            return self.user_item_evaluation_list[(u, i)]
        except AttributeError:
            train_list = self.get_train_list()
            self.user_item_evaluation_list = dict(((x[0], x[1]), x[2]) for x in train_list)
            return self.user_item_evaluation_list[(u, i)]

    def get_casual_users(self):
        """return the set of users which have 1 single vote which is 1 or 10"""
        try:
            return self.goodguys
        except AttributeError:
            user_set = self.get_user_set()
            self.goodguys = map(lambda z: z[0],
                                (filter(lambda x: len(x[1])==1 and (sum(x[1])==10 or sum(x[1])==1),
                                                       map(lambda x: (x,
                                                                      map(lambda y: self.get_evaluation(x, y),
                                                                          self.get_user_evaluation_list(x))), user_set))))
            return self.goodguys

    def get_feature_global_frequency(self, feature):
        # return the frequency of a given feature respect to all features
        feature_items_list = self.get_feature_items_list()
        if feature in feature_items_list:
            len_items = float(len(self.get_item_set()))
            feature_global_frequency = len(feature_items_list[feature])
            idf = math.log(len_items / feature_global_frequency, 10)
        else:
            idf = 0

        return idf

    def get_features_partial_frequency(self, user, feature):
        # return the frequency of a given feature among a particular user
        user_items_list = self.get_user_evaluations(user)
        user_features_list = map(lambda x: self.get_features_list(x), user_items_list)
        flattened = [item for sublist in user_features_list for item in sublist]
        if feature in flattened:
            occurrence = flattened.count(feature)
            len_items = float(len(user_items_list))
            feature_partial_frequency = occurrence
            idf = math.log(((len_items / feature_partial_frequency) * self.get_user_feature_evaluation(user, feature)), 10)
            tf_idf = occurrence * idf
            feature_prediction = tf_idf
        else:
            feature_prediction = 0

        return feature_prediction

    def get_num_users(self):
        """

        Calculating the number of distinct users that rated some items
        len(set(map(lambda x: x[0],train) + userSet))=> 15373
        last element of train=>15374

        :return:
        """
        return 15373

    def get_num_active_users(self):
        """

        Number of users that has rated some items

        :return:
        """
        active_users = self.get_active_users_set()
        return len(active_users)

    def get_num_items(self):
        try:
            return self.num_items
        except AttributeError:
            self.num_items = len(self.get_item_set())
            return self.num_items

    def get_num_ratings(self):
        """

        Calculating the number of ratings of the system

        :return:
        """
        try:
            return self.num_ratings
        except AttributeError:
            self.num_ratings = len(self.get_train_list())
            return self.num_ratings

    def get_evaluation(self, u, i):
        """

        Getting the evaluation of a specific film for a user
        :param u:
        :param i:
        :return:
        """
        try:
            return self.get_user_item_evaluation(u, i)
        except KeyError:
            return 0  # if the rating does not exist it returns zero

    def get_features_list(self, i):
        try:
            item_features_list = self.get_item_features_list()
            return item_features_list[i]
        except Exception:
            return []  # if the item does not appears it has no features

    def get_user_evaluations(self, user):
        try:
            user_evaluations_list = self.get_user_evaluations_list()
            return user_evaluations_list[user]
        except KeyError:
            return []

    def get_user_evaluations_list(self):
        try:
            return self.user_evaluations_list
        except AttributeError:
            self.generate_lists_from_train()
            return self.user_evaluations_list

    def get_items_never_seen(self):
        try:
            return self.items_never_seen
        except AttributeError:
            self.generate_lists_from_icm()
            return self.items_never_seen

    def get_item_evaluators_list(self):
        try:
            return self.item_evaluators_list
        except AttributeError:
            self.generate_lists_from_train()
            return self.item_evaluators_list

    def get_item_evaluators(self, item):
        try:
            item_evaluators_list = self.get_item_evaluators_list()
            return item_evaluators_list[item]
        except KeyError:
            return []

    def get_user_feature_evaluation(self, user, feature):
        try:
            user_feature_evaluation_list = self.get_user_feature_evaluation_list()
            return user_feature_evaluation_list[(user, feature)]
        except KeyError:
            return 0

    def get_user_feature_evaluation_count(self, user, feature):
        try:
            return self.user_feature_evaluation_count[(user, feature)]
        except KeyError:
            return 0

    def get_item_set_ordered(self):
        return sorted(self.item_set)


    def get_user_to_recommend_evaluation_count(self):
        return sorted(Counter(map(lambda x: len(self.get_user_evaluation_list(x)), self.user_set)).items(), key=lambda x: x[1],
                      reverse=True)

    def get_top_viewed(self):
        '''
        return a list of tuples (item, value) where
        item is the film
        value is the number of votes received
        '''
        try:
            return self.top_viewed
        except AttributeError:
            item_evaluators_list = self.get_item_evaluators_list()
            self.top_viewed = sorted(item_evaluators_list.items(), key=lambda x: len(x[1]), reverse=True)
            return self.top_viewed

    def populate_user_similarities(self, user):
        similarities = {}
        for userX, userY, similarity in self.get_users_similarities_list():
            if userX == user:
                    similarities[userY] = similarity
        return similarities

    def get_avg_user_rating(self, user):
        return self.avg_user_rating[user]

    def get_avg_item_rating(self, item):
        return self.avg_item_rating[item]

    def get_feature_items_list(self):
        try:
            return self.feature_items_list
        except AttributeError:
            self.generate_lists_from_icm()
            return self.feature_items_list

    def get_feature_items(self, feature):
        try:
            feature_items_list = self.get_feature_items_list()
            return feature_items_list[feature]
        except KeyError:
            return []

    def generate_lists_from_icm(self):
        """generate following lists
        - item_features_list
        - items_evaluated
        - items_never_seen
        - item_features_list
        - feature_items_list
        - feature_set"""
        item_features_list = self.get_item_features_list()
        self.items_evaluated = self.get_items_in_train()
        self.items_never_seen = set()
        self.item_features_list = {}
        self.feature_items_list = {}
        self.features_set = set()

        for elem in item_features_list:
            if not elem[0] in self.items_evaluated:
                self.items_never_seen.add(elem[0])
            if not elem[0] in self.item_features_list:
                self.item_features_list[elem[0]] = []
            self.item_features_list[elem[0]].append(elem[1])

            if not elem[1] in self.feature_items_list:
                self.feature_items_list[elem[1]] = []
            self.feature_items_list[elem[1]].append(elem[0])

            self.features_set.add(elem[1])

    def generate_lists_from_train(self):
        """generate the following lists:
        - user_features_evaluation
        - user_feature_evaluation_count
        - user_evaluations_list
        - item_evaluators_list
        - avg_usr_rating
        - avg_item_rating
        """
        self.user_feature_evaluation = {}
        self.user_feature_evaluation_count = {}
        self.user_evaluations_list = {}
        self.item_evaluators_list = {}
        self.avg_user_rating = {}
        self.avg_item_rating = {}
        count_user_rating = defaultdict(lambda: 0, {})
        count_item_rating = defaultdict(lambda: 0, {})
        train = self.get_train_list()
        for elem in train:
            u = elem[0]
            i = elem[1]
            r = elem[2]

            try:
                self.avg_user_rating[u] = (self.avg_user_rating[u] * count_user_rating[u] + float(r)) / (
                    count_user_rating[u] + 1)  # running average
            except Exception:
                self.avg_user_rating[u] = 0.0
                self.avg_user_rating[u] = (self.avg_user_rating[u] * count_user_rating[u] + float(r)) / (count_user_rating[u] + 1)
            count_user_rating[u] += 1

            try:
                self.avg_item_rating[i] = (self.avg_item_rating[i] * count_item_rating[i] + float(r)) / (
                    count_item_rating[i] + 1)  # running average
            except Exception:
                self.avg_item_rating[i] = 0.0
                self.avg_item_rating[i] = (self.avg_item_rating[i] * count_item_rating[i] + float(r)) / (
                    count_item_rating[i] + 1)  # running average
            count_item_rating[i] += 1

            try:  # need to manage the "initialization case" in which the key does not exists
                self.user_evaluations_list[u].append(i)
            except Exception:  # if the key is non-initialized, do it
                self.user_evaluations_list[u] = []
                self.user_evaluations_list[u].append(i)

            try:  # need to manage the "initialization case" in which the key does not exists
                self.item_evaluators_list[i].append(u)
            except Exception:  # if the key is non-initialized, do it
                self.item_evaluators_list[i] = []
                self.item_evaluators_list[i].append(u)

            if i in self.item_features_list:
                for f in self.item_features_list[i]:
                    try:  # need to manage the "initialization case" in which the key does not exists
                        self.user_feature_evaluation[(u, f)] = (self.user_feature_evaluation[(u, f)] *
                                                           self.user_feature_evaluation_count[
                                                               (u, f)] + float(
                            r)) / (self.user_feature_evaluation_count[(u, f)] + 1)
                        self.user_feature_evaluation_count[(u, f)] += 1
                    except Exception:  # if the key is non-initialized, do it
                        if (u, f) not in self.user_feature_evaluation:
                            self.user_feature_evaluation[(u, f)] = 0.0
                        if (u, f) not in self.user_feature_evaluation_count:
                            self.user_feature_evaluation_count[(u, f)] = 0
                        self.user_feature_evaluation[(u, f)] = (self.user_feature_evaluation[(u, f)] *
                                                           self.user_feature_evaluation_count[
                                                               (u, f)] + float(
                            r)) / (self.user_feature_evaluation_count[(u, f)] + 1)
                        self.user_feature_evaluation_count[(u, f)] += 1

    def get_user_similarities(self, userI):
        threshold = xrange(2, 10)  # threshold for films, 1 and 10 are removed to avoid the Global Effect
        similarities = []  # list in which will be stored the similarities found

        # Needed for the running Avg
        countAvgCommonMovies = defaultdict(lambda: 0, {})

        # hold the average count of the common movies of userI with all the other users
        avgCommonMovies = defaultdict(lambda: 0.0, {})

        # will hold the number of common movies found between users
        numberCommonMovies = defaultdict(lambda: 0, {})

        # will hold the pairs of lists containing rating of the two users
        evaluationLists = defaultdict(list, {})

        blacklist = [userI]  # will help to filter out user with no similarity

        shrink = {}  # This hashmap will store the shrink value relative to userIterator

        itemsUserI = self.get_user_evaluations(userI)  # get the vector of the evaluated items

        train_user_set = self.get_train_user_set()

        for userJ in train_user_set:
            itemsUserJ = self.get_user_evaluations(userJ)[
                         :]  # need to get a copy of the vector (achieved through [:]) since we are going to modify it
            ratingsUserI = []  # will contain the evaluations of User of the common items with userIterator
            ratingsUserJ = []  # will contain the evaluations of userIterato of the common items with userIterator

            for item in itemsUserI:
                if item in itemsUserJ:
                    numberCommonMovies[userJ] += 1
                    userInverseFrequency = (math.log(self.get_num_users(), 10)) - math.log(
                        len(self.get_item_evaluators(item)), 10)
                    ratingsUserI.append((self.get_evaluation(userI, item) - self.get_avg_user_rating(
                        userI)) * userInverseFrequency)
                    ratingsUserJ.append((self.get_evaluation(userJ, item) - self.get_avg_user_rating(
                        userJ)) * userInverseFrequency)

            if not (len(ratingsUserI) == 0):

                evaluationLists[userJ].append(ratingsUserI)
                evaluationLists[userJ].append(ratingsUserJ)

                avgCommonMovies[userJ] = (avgCommonMovies[userJ] * countAvgCommonMovies[userJ] + len(
                    ratingsUserJ)) / (countAvgCommonMovies[userJ] + 1)  # Running Average
                countAvgCommonMovies[userJ] += 1

                shrink[userJ] = math.fabs(math.log(float(len(ratingsUserJ)) / self.get_num_items()))
            else:
                blacklist.append(userJ)

        for userJ in train_user_set:
            if (userJ not in blacklist):
                if numberCommonMovies[userJ] >= avgCommonMovies[userJ]:

                    similarity = self.compute_pearson_user_based_correlation(userI, userJ, evaluationLists[userJ][0],
                                                                evaluationLists[userJ][1], shrink[userJ])
                else:
                    similarity = self.compute_pearson_user_based_correlation(userI, userJ, evaluationLists[userJ][0],
                                                                evaluationLists[userJ][1], shrink[userJ]) * (
                                     numberCommonMovies[userJ] / 50)  # significance weight
                similarities.append([userI, userJ, math.fabs(similarity)])
        return sorted(similarities, key=lambda x: x[2], reverse=True)[:5]

    def compute_pearson_user_based_correlation(self, u, u2, list_a, list_b, shrink):
        """

        Calculating the Pearson Correlation coefficient between two given users

        :param shrink:
        :param u:
        :param u2:
        :param list_a:
        :param list_b:
        :return:
        """
        avg_u = self.get_avg_user_rating(u)
        avg_u2 = self.get_avg_user_rating(u2)
        list_numerator_u = map(lambda x: x - avg_u, list_a)
        list_numerator_u2 = map(lambda x: x - avg_u2, list_b)
        numerator_pearson = sum([elem1 * elem2 for elem1, elem2 in zip(list_numerator_u, list_numerator_u2)])
        list_den_u = map(lambda x: x ** 2, list_numerator_u)
        list_den_u2 = map(lambda x: x ** 2, list_numerator_u2)
        denominator_pearson = math.sqrt(sum(list_den_u)) * math.sqrt(sum(list_den_u2))
        if denominator_pearson == 0:
            return 0
        pearson = numerator_pearson / (denominator_pearson + shrink)
        return pearson

    def get_item_similarities(self, itemI):
        # compute the similarity of itemI with all the remaining item in the set
        similarities = []

        for itemJ in self.get_item_set():
            if itemI != itemJ:
                usersI = self.get_item_evaluators(itemI)[:]  # take the copy of the users that evaluated itemI
                usersJ = self.get_item_evaluators(itemJ)[:]  # take the copy of the users that evaluated itemJ

                ratingsItemI = []  # will contain the evaluations of User of the common items with userIterator
                ratingsItemJ = []  # will contain the evaluations of userIterator of the common items with userIterator

                for user in usersJ:
                    if user in usersI:
                        ratingsItemI.append(self.get_evaluation(user, itemI) - self.get_avg_user_rating(user))
                        ratingsItemJ.append(self.get_evaluation(user, itemJ) - self.get_avg_user_rating(user))
                if not (len(ratingsItemI) == 0):
                    shrink = math.fabs(math.log(float(len(ratingsItemI)) / self.get_num_users()))
                    similarity = self.compute_pearson_item_based_correlation(ratingsItemI, ratingsItemJ, shrink)
                    similarities.append([itemI, itemJ, similarity])
        return sorted(similarities, key=lambda x: x[2], reverse=True)[:10]

    @staticmethod
    def compute_pearson_item_based_correlation(self, list_a, list_b, shrink):
        """

        Calculating the Pearson Correlation coefficient between two given items

        :param list_a:
        :param list_b:
        :param shrink:
        :return:
        """
        numerator_pearson = sum([elem1 * elem2 for elem1, elem2 in zip(list_a, list_b)])
        list_den_i = map(lambda x: x ** 2, list_a)
        list_den_j = map(lambda x: x ** 2, list_b)
        denominator_pearson = math.sqrt(sum(list_den_i)) * math.sqrt(sum(list_den_j))
        if denominator_pearson == 0:
            return 0
        pearson = numerator_pearson / (denominator_pearson + shrink)
        return pearson

    def compute_top_n(self, shrink):
        """

        Insert into an hashmap the total value for each
        film calculated by summing all the rating obtained through user
        rating divided by the sum of the votes + the
        variable shrink value obtained as logarithm
        of the number of votes divided for the number
        of users in the system.

        :return:
        """
        train = self.get_train_list()
        threshold = (95.0/100) * self.get_max_votes()
        summation = 0
        counter = 0
        total = {}
        last_item = train[0][1]
        for line in train:
            item = line[1]
            rating = line[2]
            if last_item != item:
                if counter >= threshold:
                    if shrink:
                        variable_shrink = math.fabs(math.log(float(counter) / self.get_num_active_users()))
                        total[last_item] = summation / float(counter + variable_shrink)
                    else:
                        total[last_item] = summation / float(counter)
                counter = 0
                summation = 0
                last_item = item
            summation += rating
            counter += 1

        # Sorting in descending order the list of items
        if shrink:
            maximum = max(total.values())
            rebalancer = 10.0/maximum
            total = map(lambda x: (x[0], x[1]*rebalancer), total.items())
            self.top_n_w_shrink = sorted(total, key=lambda x: x[1], reverse=True)
        else:
            self.top_n = sorted(total.items(), key=lambda x: x[1], reverse=True)

    def get_top_n_w_shrink(self):
        try:
            return self.top_n_w_shrink
        except AttributeError:
            self.compute_top_n(True)
            return self.top_n_w_shrink

    def get_top_n(self):
        try:
            return self.top_n
        except AttributeError:
            self.compute_top_n(False)
            return self.top_n

    def get_max_votes(self):
        try:
            return self.max_votes
        except AttributeError:
            self.max_votes = 0
            train = self.get_train_list()
            counter = 0
            last_item = train[0][1]
            for line in train:
                item = line[1]
                if last_item != item:
                    if counter > self.max_votes:
                        self.max_votes = counter
                    counter = 0
                    last_item = item
                counter += 1
            return self.max_votes
