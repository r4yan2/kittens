import math
from collections import defaultdict, Counter
import helper
import random

class Database:
    def __init__(self, test=None):
        """
        initializing the database:

        * if we are in test execution train set is splitted and test_set is generated
        :param test:
        """
        if not test == None:
            self.load_test_set(test)
            self.load_train_list(test)
        else:
            self.load_train_list()

    def get_test_set(self):
        """
        getter for the test_set, used when the engine is run in Test Mode
        if the set is not defined compute it with the appropriate method
        :return:
        """
        try:
            return self.test_set
        except AttributeError:
            self.test_set = self.compute_test_set()
            return self.test_set

    def load_test_set(self, test):
        test_set = list(helper.read("test_set" + str(test)))
        self.test_set = map(lambda x: [int(x[0]), int(x[1])], test_set)

    def get_playlists(self):
        """
        return the playlists set
        :return:
        """
        try:
            return self.playlists_set
        except AttributeError:
            playlists = self.get_train_list()
            self.playlists_set = set([playlist for playlist, track in playlists])
            return self.playlists_set

    def compute_test_set(self):
        """
        computing the test set if Testing mode is enabled:

        * REMOVE a number of lines from the train set and put them in the test set
        * every line is randomly chosen to avoid over fitting of the test set
        :return:
        """
        train = self.get_train_list()
        length = len(self.get_target_playlists()) * 5
        self.train_test = []
        playlists_with_n_tracks = set([playlist for playlist in self.get_playlists() if len(self.get_playlist_tracks(playlist)) >= 10])
        already_selected = set()
        for i in xrange(0, length):
            while True:
                line = random.randint(0, len(train) - 1)
                if line not in already_selected and train[line][0] in playlists_with_n_tracks:
                    break
            already_selected.add(line)
            self.train_test.append(train[line])
        self.train_list = helper.diff_test_set(train, self.train_test)
        self.target_tracks = map(lambda x: x[1], self.train_test)

    def get_train_list(self):
        """
        getter for the train_list

        :return:
        """
        try:
            return self.train_list
        except AttributeError:
            self.load_train_list()
            return self.train_list

    def load_train_list(self, test=None):
        if test == None:
            train_list = list(helper.read("train_final"))
            self.train_list = map(lambda x: [int(x[0]), int(x[1])], train_list[1:])
        else:
            train_list = list(helper.read("train_set"+str(test)))
            self.train_list = map(lambda x: [int(x[0]), int(x[1])], train_list)

    def get_tag_playlists_map(self):
        """

        :return:
        """
        try:
            return self.tag_playlists_map
        except AttributeError:
            self.tag_playlists_map = defaultdict(lambda: [], {})
            for playlist in self.get_playlists():
                tracks = self.get_playlist_tracks(playlist)  # get already included tracks
                tags = []
                [tags.extend(self.get_track_tags(track)) for track in tracks] # get the tags from the active tracks
                tags_set = set(tags)
                for tag in tags_set:
                    self.tag_playlists_map[tag].append(playlist)
            return self.tag_playlists_map

    def get_tag_idf(self, tag):
        """

        :return:
        """
        tag_playlist_map = self.get_tag_playlists_map()
        playlist_tags_included = tag_playlist_map[tag]
        num_idf = len(playlist_tags_included)
        den_idf = len(self.get_playlists())
        try:
            idf = math.log(num_idf/float(den_idf), 10)
        except ValueError:
            idf = 0
        return idf

    def get_playlist_final(self):
        """
        the initialization of the playlist final csv
        :return:
        """
        try:
            return self.playlist_final

        except AttributeError:

            playlist_list = list(helper.read("playlists_final"))
            result = {}
            for playlist in playlist_list[1:]:
                created_at = int(playlist[0])
                playlist_id = int(playlist[1])
                title = eval(playlist[2])
                numtracks = int(playlist[3])
                duration = int(playlist[4])
                owner = int(playlist[5])

                result[playlist_id]= [created_at, title, numtracks, duration, owner]
            self.playlist_final = result
            return self.playlist_final

    def get_target_playlists(self):
        """
        getter for the target playlists for which recommend tracks in target_tracks
        if the target_playlists does not exists it create the list from the corresponding csv
        :return:
        """
        try:
            return self.target_playlists
        except AttributeError:
            self.compute_target_playlists()
            return self.target_playlists

    def compute_target_playlists(self):
        """
        return the list from the csv of the target_playlist, with the first row removed
        :return:
        """
        target_playlists = list(helper.read("target_playlists"))
        self.target_playlists = map(lambda x: int(x[0]), target_playlists[1:])

    def get_target_tracks(self):
        """
        getter for the tracks to be recommended
        if the target_tracks does not exists it create the list from the corresponding csv
        :return:
        """
        try:
            return self.target_tracks
        except AttributeError:
            target_tracks = list(helper.read("target_tracks"))
            self.target_tracks = map(lambda x: int(x[0]), target_tracks[1:])
            return self.target_tracks

    def get_users_similarities_list(self):
        try:
            return self.user_similarities_list
        except AttributeError:
            user_similarities_list = list(self.user_similarities_csv)
            self.user_similarities_list = map(lambda x: map(float, x), user_similarities_list)
            return self.user_similarities_list

    def get_tracks_map(self):
        """
        getter for tracks list
        :return:
        """
        try:
            return self.tracks_map
        except AttributeError:
            self.tracks_map = self.compute_tracks_map()
            return self.tracks_map

    def compute_tracks_map(self):
        """
        parse tracks_final.csv dividing all field into the corresponding part of the list
        :return:
        """
        tracks = list(helper.read("tracks_final"))
        result = {}
        for track in tracks[1:]:
            track_id = int(track[0])
            artist_id = int(track[1])
            duration = int(track[2])
            if duration == None:
                continue
            try:
                playcount = float(track[3]) # yes, PLAYCOUNT is memorized as a floating point
            except ValueError:
                playcount = 0.0
            album = eval(track[4])
            try:
                album = int(album[0]) # yes again, album is memorized as a list, even if no track have more than 1 album
            except TypeError:
                album = 0
            except IndexError:
                album = 0
            tags = eval(track[5]) # evaluation of the tags list
            result[track_id]= [artist_id, duration, playcount, album, tags]
        return result

    def get_playlist_user_tracks(self, playlist):
        """
        return the tracks listened by a user
        :param playlist:
        :return:
        """
        playlist_final = self.get_playlist_final()
        owned_by = playlist_final[playlist][4]

        owner_playlist = self.get_owner_playlists()
        playlist_list = owner_playlist[owned_by]

        tracks_listened = []
        for playlist in playlist_list:
            tracks = self.get_playlist_tracks(playlist)
            tracks_listened.extend(tracks)

        return set(tracks_listened)

    def get_owner_playlists(self):
        """

        :return:
        """
        try:
            return self.owner_playlist

        except AttributeError:
            playlists = list(helper.read("playlists_final"))
            self.owner_playlist = defaultdict(lambda: [], {})
            for owner in playlists[1:]:
                owned_by = int(owner[5])
                playlist_id = int(owner[1])
                self.owner_playlist[owned_by].extend([playlist_id])

            return self.owner_playlist

    def get_track_duration(self, track):
        """
        return the length of a given track
        :param track:
        :return:
        """
        tracks_map = self.get_tracks_map()
        return tracks_map[track][1]

    def get_playlist_avg_duration(self, playlist):
        """
        return the avg length of the tracks in the given playlist
        :param playlist:
        :return:
        """
        tracks = self.get_playlist_tracks(playlist)
        length = []
        for track in tracks:
            track_length = self.get_track_duration(track)
            if track_length <= 0:
                continue
            length.append(track_length)
        try:
            return float(sum(length))/len(length)
        except ZeroDivisionError:
            return 0.0


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

    def get_tracks_set(self):
        """
        gettter for the tracks set.
        get_tracks and get_tracks_set are different because the former return the dataset taken from tracks_final
        the latter return a set containing the id of all the tracks in the dataset
        :return:
        """
        try:
            return self.tracks_set
        except AttributeError:
            train = self.get_train_list()
            tracks = self.get_tracks()
            self.tracks_set = set(map(lambda x: x[0], tracks) + map(lambda x: x[1], train))
            return self.tracks_set

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

    def get_num_users(self):
        """

        Calculating the number of distinct users that rated some items
        len(set(map(lambda x: x[0],train) + userSet))=> 15373
        last element of train=>15374

        :return:
        """
        try:
            return self.num_users
        except AttributeError:
            playlists = self.get_playlists()
            self.num_users = len(set(map(lambda x: x[5], playlists)))
            return self.num_users


    def get_num_active_users(self):
        """

        Number of users that has rated some items

        :return:
        """
        active_users = self.get_active_users_set()
        return len(active_users)

    def get_num_tracks(self):
        """
        getter of the total of tracks in the dataset
        :return:
        """
        try:
            return self.num_tracks
        except AttributeError:
            self.num_tracks = len(self.get_tracks())
            return self.num_tracks

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

    def get_track_tags(self, track):
        """
        compute tags from tracks_final given a specific track

        :param track:
        :return:
        """
        track_map = self.get_tracks_map()
        try:
            return track_map[track][4]
        except LookupError:
            return []


    def get_from_list(self, track, tracks):
        """
        binary search hand implemented
        :param track:
        :param tracks:
        :return:
        """
        lenght = len(tracks)
        half = lenght/2
        quarter = half/2
        halfquarter = half + quarter

        if track < tracks[half][0]:
            if track < tracks[quarter][0]:
                for line in tracks[:quarter]:
                    if track == line[0]:
                        return line
            else:
                for line in tracks[quarter:half]:
                    if track == line[0]:
                        return line
        else:
            if track < tracks[halfquarter][0]:
                for line in tracks[half:halfquarter]:
                    if track == line[0]:
                        return line
            else:
                for line in tracks[halfquarter:]:
                    if track == line[0]:
                        return line

    def get_track_tags_map(self):
        """
        compute hashmap to store track -> [tag1, tag2, ...] for fast retrieval
        :return:
        """

        try:
            return self.track_tags_map
        except AttributeError:
            self.track_tags_map = {}
            tracks = self.get_tracks()
            for track in tracks:
                id = track[0]
                tags = track[5]
                self.track_tags_map[id] = tags
            return self.track_tags_map

    def get_playlist_tracks(self, target_playlist):
        """
        return all the tracks into the specified playlist
        :param playlist:
        :return:
        """
        try:
            return self.playlist_tracks_map[target_playlist]
        except AttributeError:
            self.playlist_tracks_map = defaultdict(lambda: [], {})
            train_list = self.get_train_list()
            [self.playlist_tracks_map[playlist].append(track) for playlist, track in train_list]
            return self.playlist_tracks_map[target_playlist]

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

    def get_track_playlists(self, track):
        """
        return all the playlists which includes the specified track
        :param track:
        :return:
        """
        try:
            return self.track_playlists_map[track]
        except AttributeError:
            track_playlists_map = self.get_track_playlists_map()
            return track_playlists_map[track]

    def get_track_playlists_map(self):
        """
        return the hashmap which links track and playlists which includes the track
        :return:
        """
        try:
            return self.track_playlists_map
        except AttributeError:
            self.track_playlists_map = defaultdict(lambda: [], {})
            playlists = self.get_train_list()
            for row in playlists:
                playlist = row[0]
                track = row[1]
                self.track_playlists_map[track].append(playlist)
            return self.track_playlists_map

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

    def get_top_included(self):
        '''
        return a list of tuples (track, value) where
        value is the number of featured playlists
        '''
        try:
            return self.top_included
        except AttributeError:
            track_playlists_map = self.get_track_playlists_map()
            track_playlists = track_playlists_map.items()
            self.top_included = sorted(map(lambda x: [x[0], len(x[1])], track_playlists), key=lambda x: x[1], reverse=True)
            return self.top_included

    def get_top_listened(self):
        '''
        return a list of tuples (track, value) where
        value is the number of times played
        '''
        try:
            return self.top_listened
        except AttributeError:
            tracks = self.get_tracks()
            self.top_listened = sorted(map(lambda x: [x[0], x[3]], tracks), key=lambda x: x[1], reverse=True)
            return self.top_listened

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
        # Higher threshold give high accuracy...but over 85% does not have more that 10 items
        threshold = (50.0/100) * self.get_max_votes()
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
            # rebalance need to put back on scale the list
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
