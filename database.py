import math
from collections import defaultdict, Counter
import helper
import random
import logging
from operator import itemgetter

class Database:
    def __init__(self, test, individual=False, coefficient=False, extended=True, tag_whitelist=True, title_whitelist=False, track_whitelist=False):
        """
        if we are in test execution train_set, test_set are loaded transparently,
        otherwise the normal dataset are loaded

        :param test: istance, 0 for recommendations, 4-6 for test_set
        :param individual: if set specify an individual to use (DEAP)
        :param tags: extended mean to extend default tags with artist and album
        :param whitelist: if whitelist have to be loaded
        :return: the initialized object
        """

        if individual:
            fp = open(individual, "rb")
            self.individual_filename = individual
            self.individual_coefficient = helper.parseList(fp.readline(), float)
            self.individual = True
            fp.close()
        else:
            self.individual = False
            self.individual_coefficient = False

        self.extended = extended
        self.tag_whitelist = tag_whitelist
        self.title_whitelist = title_whitelist
        self.track_whitelist = track_whitelist
        self.test = test

        logging.debug("Loading database with istance %i..." % (test))

        if test > 0:
            self.load_test_set(test)
            self.load_train_list(test)
        else:
            self.load_train_list()
        self.get_playlist_final()

    def get_individual_state(self):
        """
        """
        return self.individual

    def set_individual_state(self, state):
        """
        """
        if state:
            if not self.individual_filename:
                raise ValueError("Will not activate individual because no individual was set upon start!")
        self.individual = state

    def get_individual_filename(self):
        """
        """
        return self.individual_filename

    def get_target_playlists_tracks_set(self):
        """
        Compute the set of all the tracks present in playlists

        :return: a set of tracks
        """
        try:
            return self.target_playlists_tracks_set
        except AttributeError:
            target_playlists = self.get_target_playlists()
            target_playlists_tracks_set = set([track for playlist in target_playlists for track in self.get_playlist_tracks(playlist)])
            target_tracks = self.get_target_tracks()
            self.target_playlists_tracks_set = target_playlists_tracks_set
            return self.target_playlists_tracks_set

    def get_test_set(self):
        """
        getter for the test_set, used when the engine is run in Test Mode
        if the set is not defined compute it with the appropriate method

        :return: list of playlist,track corresponding to the slice of train taken
        """
        return self.test_set

    def load_test_set(self, test):
        """
        Loader for the test set

        :param test: integer which indicate the specific test istance
        :return: None
        """
        test_set = helper.read("test_set" + str(test))

        self.test_set = [[int(playlist), int(track)] for playlist, track in test_set]
        self.target_playlists = set([playlist for playlist, track in self.test_set])
        self.target_tracks = set([track for playlist, track in self.test_set])

    def get_playlists(self):
        """
        getter for the playlists set IN TRAIN

        :return: set of playlists
        """
        try:
            return self.playlists_set
        except AttributeError:
            playlists = self.get_train_list()
            self.playlists_set = set([playlist for playlist, track in playlists])
            return self.playlists_set

    def compute_test_set_v1(self):
        """
        Computing the test set if Testing mode is enabled:

        * REMOVE a number of lines from the train set and put them in the test set
        * every line is randomly chosen to avoid over fitting of the test set

        :return: None
        """
        train = self.get_train_list()
        playlist_length = 10000
        test_length = 10*len(train)/100.0
        self.test_set = []
        playlists_with_n_tracks = [playlist for playlist in self.get_playlists() if len(self.get_playlist_tracks(playlist)) >= 10]
        already_selected = set()
        already_selected_playlists = set()
        while True:
            if playlist_length <= 0:
                break
            playlist = random.choice(playlists_with_n_tracks)
            if playlist not in already_selected_playlists:
                already_selected_playlists.add(playlist)
                playlist_length -= 1
        for playlist in already_selected_playlists:
            tracks = self.get_playlist_tracks(playlist)
            already_selected_tracks = set()
            tracks_length = 5
            while True:
                if tracks_length <= 0:
                    break
                track = random.choice(tracks)
                if track not in already_selected_tracks:
                    already_selected_tracks.add(track)
                    tracks_length -= 1
                    self.test_set.append([playlist, track])

        self.train_list = helper.diff_test_set(train, self.test_set)

    def compute_test_set_v2(self, percentage=10):
        """
        Computing the test set if Testing mode is enabled:

        * REMOVE a number of lines from the train set and put them in the test set
        * every line is randomly chosen to avoid over fitting of the test set

        :return: None
        """
        train = self.get_train_list()
        test = []
        train_length = len(train)
        test_length = int(train_length*percentage/100.0)
        already_selected = set()
        while True:
            if test_length > 0:
                choice = random.randint(0,train_length-1)
                if choice not in already_selected and len(self.get_playlist_tracks(train[choice][0])) >= 8:
                    test.append(train[choice])
                    test_length -= 1
                    already_selected.add(choice)
            else:
                break
        self.train_list = helper.diff_test_set(train, test)
        self.test_set = test

    def get_user_tracks(self, user):
        """
        Getter for the tracks of the user

        :param user: the user for which we need the tracks
        :return: the list of tracks
        """
        owner_playlists = self.get_owner_playlists()
        user_tracks = [track for playlist in owner_playlists[user] for track in self.get_playlist_tracks(playlist)]
        return user_tracks

    def compute_content_playlists_similarity(self, playlist_a, knn=200, single_tag=0, title_flag=0, tag_flag=0, track_flag=1, tracks_knn=300, coefficient="jaccard", return_values=False, only_one=False, cumulative_track_value=False):
        """
        This method compute the neighborhood for a given playlists by using content or collaborative techniques

        :param playlist_a: given playlist
        :param knn: cardinality of the neighborhood(in playlists) ***USE tracks_knn INSTEAD***
        :param title_flag: specify if title have to be used
        :param tag_flag: specify if tag have to be used
        :param track_flag: specify if track have to be used
        :param tracks_knn: specify the quantity of neighbor track to return
        :param coefficient: specify which coefficient to use
        :param return_values: if computed value of neighborhoodness have to be given back with playlists
        :param only_one: if active only playlists with similarity 1 are considered
        :param cumulative_track_value: if active the tracks neighborhood is build accumulating values from playlists
        :return: neighborhood
        """
        playlists = self.get_playlists()

        if single_tag:
            single_tag = self.get_playlist_significative_tag(playlist_a)
            if single_tag == -1:
                return []
            else:
                neighborhood = (playlist_b for playlist_b in playlists if self.get_playlist_numtracks(playlist_b) > 10 and self.get_playlist_significative_tag(playlist_b) == single_tag)
                if not (title_flag or tag_flag or track_flag):
                    target_tracks = self.get_target_tracks()
                    return target_tracks.intersection([track for playlist in neighborhood for track in self.get_playlist_tracks(playlist)])
                else:
                    playlists = neighborhood
        if title_flag or tag_flag or track_flag:
            playlist_a_tags = []
            playlist_a_titles = []
            playlist_a_tracks = []
            tf_idf_playlist_a_tag = []
            tf_idf_playlist_a_title = []
            tf_idf_playlist_a_track = []


            if tag_flag:
                playlist_a_tags = [tag for track in self.get_playlist_tracks(playlist_a) for tag in self.get_track_tags(track)]
                playlist_a_tags_unique = list(set(playlist_a_tags))

            if title_flag:
                playlist_a_titles = self.get_playlist_titles(playlist_a)

            if track_flag:
                playlist_a_tracks = self.get_playlist_tracks(playlist_a)

            if not (len(playlist_a_tags) and tag_flag or len(playlist_a_titles) and title_flag or len(playlist_a_tracks) and track_flag):
                raise ValueError("cannot generate neighborhood for %s" % playlist_a)

            if coefficient == "cosine":

                if tag_flag:
                    tf_idf_playlist_a_tag = [playlist_a_tags.count(tag) for tag in playlist_a_tags_unique]

                if title_flag:
                    tf_idf_playlist_a_title = [self.get_title_idf(title) for title in playlist_a_titles]

                if track_flag:
                    tf_idf_playlist_a_track = [self.get_track_idf(track) for track in playlist_a_tracks]

                tf_idf_playlist_a = tf_idf_playlist_a_tag + tf_idf_playlist_a_title + tf_idf_playlist_a_track

            neighborhood = []

            for playlist_b in playlists:

                if self.get_playlist_numtracks(playlist_b) < 10 or playlist_b == playlist_a:
                    continue

                playlist_b_tags = []
                playlist_b_titles = []
                playlist_b_tracks = []
                tf_idf_playlist_b_tag = []
                tf_idf_playlist_b_title = []
                tf_idf_playlist_b_track = []

                if tag_flag:
                    playlist_b_tags = [tag for track in self.get_playlist_tracks(playlist_b) for tag in self.get_track_tags(track)]
                    playlist_b_tags_unique = list(set(playlist_b_tags))

                if title_flag:
                    playlist_b_titles = self.get_playlist_titles(playlist_b)

                if track_flag:
                    playlist_b_tracks = self.get_playlist_tracks(playlist_b)

                if not (len(playlist_b_tags) and tag_flag or len(playlist_b_titles) and title_flag or len(playlist_b_tracks) and track_flag):
                    continue

                if coefficient == "cosine":

                    if tag_flag:
                        tf_idf_playlist_b_tag = [playlist_b_tags.count(tag) for tag in playlist_b_tags_unique]

                    if title_flag:
                        tf_idf_playlist_b_title = [self.get_title_idf(title) for title in playlist_b_titles]

                    if track_flag:
                        tf_idf_playlist_b_track = [self.get_track_idf(track) for track in playlist_b_tracks]

                tf_idf_playlist_b = tf_idf_playlist_b_tag + tf_idf_playlist_b_title + tf_idf_playlist_b_track

                num_cosine_sim_tag = 0
                num_cosine_sim_title = 0
                num_cosine_sim_track = 0

                if coefficient == "cosine":
                    if tag_flag:
                        num_cosine_sim_tag = sum([tf_idf_playlist_a_tag[playlist_a_tags_unique.index(tag)] * tf_idf_playlist_b_tag[playlist_b_tags_unique.index(tag)] for tag in playlist_b_tags_unique if tag in playlist_a_tags_unique])

                    if title_flag:
                        num_cosine_sim_title = sum([tf_idf_playlist_a_title[playlist_a_titles.index(title)] * tf_idf_playlist_b_title[playlist_b_titles.index(title)] for title in playlist_b_titles if title in playlist_a_titles])

                    if track_flag:
                        num_cosine_sim_track = sum([tf_idf_playlist_a_track[playlist_a_tracks.index(track)] * tf_idf_playlist_b_track[playlist_b_tracks.index(track)] for track in playlist_b_tracks if track in playlist_a_tracks])

                    num_cosine_sim = num_cosine_sim_tag + num_cosine_sim_title + num_cosine_sim_track
                    den_cosine_sim = math.sqrt(sum([i ** 2 for i in tf_idf_playlist_a])) * math.sqrt(sum([i ** 2 for i in tf_idf_playlist_b]))

                    try:
                        similarity = num_cosine_sim / den_cosine_sim
                    except ZeroDivisionError:
                        continue

                elif coefficient == "jaccard":
                    similarity_title = 1
                    similarity_track = 1
                    similarity_tag = 1
                    if title_flag:
                        similarity_title = helper.jaccard(playlist_a_titles, playlist_b_titles)
                    if track_flag:

                        similarity_track = helper.jaccard(playlist_a_tracks, playlist_b_tracks)

                    if tag_flag:
                        similarity_tag = helper.jaccard(playlist_a_tags, playlist_b_tags)
                    similarity = similarity_tag * similarity_title * similarity_track

                if similarity > 0:
                    neighborhood.append([playlist_b, similarity])
        if only_one:
            neighborhood = [[playlist, value] for playlist, value in neighborhood if value == 1]
        else:
            neighborhood = sorted(neighborhood, key=itemgetter(1), reverse=True)

        if not tracks_knn:
            if return_values:
                return neighborhood[0:knn]
            else:
                return [playlist for playlist, _ in neighborhood[0:knn]]

        if isinstance(tracks_knn,int):
            iterator = 0
            target_tracks = self.get_target_tracks()

            if cumulative_track_value:
                tracks = {}
                for playlist, value in neighborhood:
                    for t in self.get_playlist_tracks(playlist):
                        if t in target_tracks:
                            try:
                                tracks[t] += value
                            except KeyError:
                                tracks[t] = value
                tracks = sorted(tracks.items(), key=itemgetter(1), reverse=True)
                return [track for track, _ in tracks[:tracks_knn]]

            if return_values:
                tracks = {}
                while len(tracks) < tracks_knn:
                    for t in target_tracks.intersection(self.get_playlist_tracks(neighborhood[iterator][0])).difference(playlist_a_tracks):
                        try:
                            tracks[t] += neighborhood[iterator][1]
                        except KeyError:
                            tracks[t] = neighborhood[iterator][1]
                    iterator += 1
            else:
                tracks = set()
                while len(tracks) < tracks_knn:
                    try:
                        tracks.update(target_tracks.intersection(self.get_playlist_tracks(neighborhood[iterator][0])).difference(playlist_a_tracks))
                        iterator += 1
                    except IndexError as e:
                        logging.debug("Hit an Index Error when selecting tracks for %i\nIndex: %i Error:%s" % (playlist_a, iterator, e))
                        return tracks

            return tracks
        elif isinstance(tracks_knn, list):
            target_tracks = self.get_target_tracks()
            tracks_knn.sort()
            to_return = []
            if cumulative_track_value:
                tracks = {}
                for playlist, value in neighborhood:
                    for t in self.get_playlist_tracks(playlist):
                        if t in target_tracks:
                            try:
                                tracks[t] += value
                            except KeyError:
                                tracks[t] = value
                tracks = sorted(tracks.items(), key=itemgetter(1), reverse=True)
                for knn in tracks_knn:
                    to_return.append([track for track, _ in tracks[:knn]])
            else:
                for knn in tracks_knn:
                    iterator = 0
                    if return_values:
                        tracks = {}
                    else:
                        tracks = set()
                    while len(tracks) < knn:
                        if return_values:
                            for t in target_tracks.intersection(self.get_playlist_tracks(neighborhood[iterator][0])).difference(playlist_a_tracks):
                                if t not in tracks:
                                    tracks[t] = neighborhood[iterator][1]
                            iterator += 1
                        else:
                            try:
                                tracks.update(target_tracks.intersection(self.get_playlist_tracks(neighborhood[iterator][0])).difference(playlist_a_tracks))
                                iterator += 1
                            except IndexError as e:
                                break
                    to_return.append(tracks)
            return to_return

    def compute_collaborative_playlists_similarity(self, playlist, knn=50, tracks_knn=300, coefficient="jaccard", values="None", target_tracks="include"):
        """
        This method computes the similarities between playlists based on the included tracks.
        Various coefficient can be used (jaccard, map, cosine, pearson)
        NB: the best is jaccard

        :param playlist: active_playlist
        :param knn: cardinality of the neighborhood *** USE tracks_knn INSTEAD ***
        :param tracks_knn: an alternate way of returning results, if specified return the knn best matching tracks
        :param coefficient: coefficient to use
        :param values: if None return only the playlists, if all return also the values
        :param target_tracks: default include only target_tracks in tracks_knn
        :return: list of playlists
        """
        playlists = self.get_playlists()
        tracks_playlist = set(self.get_playlist_tracks(playlist))
        tracks_playlist_length = len(tracks_playlist)
        created_at_active = self.get_created_at_playlist(playlist)
        similarities = []

        for playlist_b in playlists:
            '''
            created_at = self.get_created_at_playlist(playlist_b)
            #if math.fabs(created_at_active - created_at) > (60 * 60 * 24 * 365 * 3) or
            if self.get_playlist_numtracks(playlist_b) < 10:
                continue
            '''

            tracks_playlist_b = set(self.get_playlist_tracks(playlist_b))
            tracks_playlist_b_length = len(tracks_playlist_b)



            if coefficient == "jaccard":
                jaccard = helper.jaccard(tracks_playlist, tracks_playlist_b)
                if jaccard > 0:
                    similarities.append([playlist_b, jaccard])

            elif coefficient == "cosine":
                num_cosine_sim = sum([float(track in tracks_playlist) for track in tracks_playlist_b])

                den_cosine_sim = math.sqrt(len(tracks_playlist)) * math.sqrt(
                len(tracks_playlist_b))

                try:
                    cosine_sim = num_cosine_sim / den_cosine_sim
                except ZeroDivisionError:
                    cosine_sim = 0

                similarities.append([playlist_b, cosine_sim])
            elif coefficient == "pearson":

                matched = sum([float(track in tracks_playlist) for track in tracks_playlist_b])

                not_matched_a = tracks_playlist_length - matched
                not_matched_b = tracks_playlist_b_length - matched
                mean_playlist_a = tracks_playlist_length / self.get_num_tracks()
                mean_playlist_b = tracks_playlist_b_length / self.get_num_tracks()

                numerator = sum([matched * (1.0 - mean_playlist_a) * (1.0 - mean_playlist_b), not_matched_a * (1.0 - mean_playlist_a) * (0.0 - mean_playlist_b), not_matched_b * (0.0 - mean_playlist_a) * (1.0 - mean_playlist_b), (self.get_num_tracks() - not_matched_a - not_matched_b - matched) * (0.0 - mean_playlist_a) * (0.0 - mean_playlist_b)])

                denominator = math.sqrt(sum([tracks_playlist_length * (1.0 - mean_playlist_a), (self.get_num_tracks() - tracks_playlist_length) * (0.0 - mean_playlist_a)])) * math.sqrt(sum([tracks_playlist_b_length * (1.0 - mean_playlist_b), (self.get_num_tracks() - tracks_playlist_length) * (0.0 - mean_playlist_b)]))

                pearson = numerator/denominator

                similarities.append([playlist_b, pearson])

        similarities.sort(key=itemgetter(1), reverse=True)
        if tracks_knn == None:
            if values == "None":
                return  [playlist for playlist, value in similarities[0:knn]]
            elif values == "all":
                return similarities[0:knn]
        else:
            tracks = set()
            iterator = 0
            if target_tracks != "include":
                while len(tracks) < tracks_knn:
                    tracks = tracks.union(self.get_playlist_tracks(similarities[iterator][0]))
                    iterator += 1
            else:
                target_tracks = self.get_target_tracks()
                while len(tracks) < tracks_knn:
                    try:
                        tracks.update(target_tracks.intersection(self.get_playlist_tracks(similarities[iterator][0])).difference(tracks_playlist))
                    except IndexError as e:
                        logging.debug("Hit an Index Error when selecting tracks for %i\nIndex: %i Error:%s" % (playlist, iterator, e))
                        return tracks
                    iterator += 1
        return tracks

    def get_min_max_playlists(self):
        """
        Getter for the min and max values of playlists identifier
        :return: tuple of int
        """
        try:
            return self.min_playlist, self.max_playlist
        except AttributeError:
            playlists = self.get_playlists()
            self.min_playlist = min(playlists)
            self.max_playlist = max(playlists)
            return self.min_playlist, self.max_playlist

    def get_min_max_tracks(self):
        """
        Getter for the min and max values of tracks identifier
        :return: tuple of int
        """
        try:
            return self.min_track, self.max_track
        except AttributeError:
            tracks = self.get_tracks()
            self.min_track = min(tracks)
            self.max_track = max(tracks)
            return self.min_track, self.max_track


    def get_taxonomy_value(self, i, j):
        """
        Get Taxonomy comparison result from 2 tracks
        :param i: track
        :param j: track
        :return: taxonomy value (beware: possibility of 0)
        """

        tags_i = self.get_track_tags(i)
        tags_j = self.get_track_tags(j)
        taxonomy = sum(int(feat in tags_j) for feat in tags_i) # 1 for every tag/artist/album which match

        return taxonomy


    def get_user_based_collaborative_filtering(self, active_playlist, knn=20, coefficient="jaccard"):
        """
        This method is created in order to provide a CF via users.
        Calculate the most K similar users to the active one by jaccard index
        Note that the given tracks are NOT exclusively target tracks

        :param active_playlist: active playlist
        :param knn: cardinality of the neighborhood
        :param coefficient: coefficient to use
        :return: list of tracks of the knn similar users
        """

        active_tracks = self.get_playlist_user_tracks(active_playlist)
        active_tracks_counter = Counter(active_tracks)
        already_scanned_user = set()
        playlists = self.get_playlists()
        created_at_active = self.get_created_at_playlist(active_playlist)
        target_tracks = self.get_target_tracks()

        neighborhood = []
        for playlist in playlists:
            created_at = self.get_created_at_playlist(playlist)
            if not math.fabs(created_at_active - created_at) < (60 * 60 * 24 * 365 * 3):
                continue
            user = self.get_playlist_user(playlist)
            if user in already_scanned_user:
                continue
            already_scanned_user.add(user)
            tracks = self.get_playlist_user_tracks(playlist)
            tracks_counter = Counter(tracks)

            if coefficient == "cosine":
                numerator = sum([active_tracks_counter[track] * tracks_counter[track] for track in tracks if track in active_tracks])
                denominator = math.sqrt(sum([elem * elem for elem in active_tracks_counter.values()])) * math.sqrt(sum([elem * elem for elem in tracks_counter.values()]))
                try:
                    similarity = numerator / denominator
                except ZeroDivisionError:
                    continue

            elif coefficient == "jaccard":
                similarity = helper.jaccard(active_tracks, tracks)

            neighborhood.append([target_tracks.intersection(tracks), similarity])
        neighborhood.sort(key=itemgetter(1), reverse=True)
        return [track for tracks, value in neighborhood[0:knn] for track in tracks]

    def get_knn_track_similarities(self, active_track, knn=150):
        """
        Getter for the most similar track to the passed one

        :param active_track: active_track
        :param knn: cardinality of the neighborhood
        :return: list of similar tracks
        """
        similarities_list = []
        try:
            similarities_map = self.similarities_map
        except AttributeError: # similarities_map need to be initialized
            self.get_item_similarities_alt(0,0)
            similarities_map = self.similarities_map
        for (track_a, track_b) in similarities_map.iterkeys():
            if active_track == track_a:
                similarities_list.append([track_b, similarities_map[(active_track, track_b)]])
            elif active_track == track_b:
                similarities_list.append([track_a, similarities_map[(track_a, active_track)]])
        return sorted(similarities_list, key=itemgetter(1), reverse=True)[0:knn]

    def get_item_similarities(self, i, j):
        """
        This method parse the item similairities csv and returns the similarity
         between tracks i and j

        :param i: track i int
        :param j: track j int
        :return: similarity between i and j float
        """
        try:
            return self.similarities_map[(i,j)]
        except KeyError:
            return 0.0
            try:
                return self.similarities_map[(j,i)]
            except KeyError:
                return 0.0
        except AttributeError:
            similarities = helper.read("STAI TOCCANDO LA FUNZIONE SBAGLIATA!"+str(self.test), ",")
            self.similarities_map = {}
            for (x,y, value) in similarities:
                x = int(x)
                y = int(y)
                value = float(value)
                self.similarities_map[(x,y)] = value
            try:
                return self.similarities_map[(i,j)]
            except KeyError:
                return 0.0
                return self.similarities_map[(j,i)]

    def get_num_interactions(self):
        """
        return the number of lines in train

        :return: integer
        """
        return self.num_interactions

    def init_item_similarities_epoch(self):
        """
        Init the similarities map used later for the epoch iteration method
        """
        self.similarities_map = defaultdict(lambda: defaultdict(lambda: 0.0),{})

    def get_item_similarities_epoch(self, i, j):
        """
        """
        if i < j:
            return self.similarities_map[i][j]
        elif i > j:
            return self.similarities_map[j][i]
        else:
            return 0.0

    def get_item_similarities_alt(self, i, j):
        """
        This method parse the item similairities csv and returns the similarity
         between tracks i and j

        :param i: track i int
        :param j: track j int
        :return: similarity between i and j float
        """
        try:
            return self.similarities_map[i][j]
        except KeyError:
            try:
                return self.similarities_map[j][i]
            except KeyError:
                return 0.0
        except AttributeError:
            similarities = helper.read("item-item-similarities" + str(self.test), ",")
            self.similarities_map = {}
            old_x = 0
            for (x, y, value) in similarities:
                x = int(x)
                y = int(y)
                value = float(value)
                if old_x != x:
                    limit = 200
                    self.similarities_map[x] = {}
                    self.similarities_map[x][y] = value
                    old_x = x
                elif limit > 0:
                    self.similarities_map[x][y] = value
                    limit -= 1

            try:
                return self.similarities_map[i][j]
            except KeyError:
                try:
                    return self.similarities_map[j][i]
                except KeyError:
                    return 0.0

    def get_tag_similarities(self, i, j):
        """
        This method parse the item similairities csv and returns the similarity
         between tags i and j

        :param i: tag i int
        :param j: tag j int
        :return: similarity between i and j float
        """
        try:
            return self.similarities_map_tags[i][j]
        except KeyError:
            try:
                return self.similarities_map_tags[j][i]
            except KeyError:
                return 0.0
        except AttributeError:
            similarities = helper.read("tag-tag-similarities" + str(self.test), ",")
            self.similarities_map_tags = {}
            old_x = 0
            for (x, y, value) in similarities:
                x = int(x)
                y = int(y)
                value = float(value)
                if old_x != x:
                    limit = 200
                    self.similarities_map_tags[x] = {}
                    self.similarities_map_tags[x][y] = value
                    old_x = x
                elif limit > 0:
                    self.similarities_map_tags[x][y] = value
                    limit -= 1

            try:
                return self.similarities_map_tags[i][j]
            except KeyError:
                try:
                    return self.similarities_map_tags[j][i]
                except KeyError:
                    return 0.0

    def get_similar_artist(self, a, knn=None, return_values=True):
        """
        """
        self.get_artist_similarities(0,0)
        try:
            result = self.similarities_map_artists[a]
            if knn:
                result = sorted(result.items(), key=itemgetter(1), reverse=True)[0:knn]
            else:
                result = result.items()
            if return_values:
                return result
            else:
                return [elem[0] for elem in result]
        except KeyError:
            return []

    def get_similar_user(self, u, knn=None, return_values=True):
        """
        """
        self.get_user_similarities(0,0)
        try:
            result = self.similarities_map_users[u]
            if knn:
                result = sorted(result.items(), key=itemgetter(1), reverse=True)[0:knn]
            else:
                result = result.items()
            if return_values:
                return result
            else:
                return [elem[0] for elem in result]
        except KeyError:
            return []

    def get_similar_tracks(self, t, knn=None, return_values=True):
        """
        """
        self.get_item_similarities_alt(0,0)
        try:
            result = self.similarities_map[t]
            if knn:
                result = sorted(result.items(), key=itemgetter(1), reverse=True)[0:knn]
            else:
                result = result.items()
            if return_values:
                return result
            else:
                return [elem[0] for elem in result]
        except KeyError:
            return []

    def get_similar_tracks_on_tags(self, t, knn=None, return_values=True):
        """
        """
        self.get_tag_similarities(0,0)
        try:
            result = self.similarities_map_tags[t]
            if knn:
                result = sorted(result.items(), key=itemgetter(1), reverse=True)[0:knn]
            else:
                result = result.items()
            if return_values:
                return result
            else:
                return [elem[0] for elem in result]
        except KeyError:
            return []



    def get_artist_similarities(self, i, j):
        """
        This method parse the item similairities csv and returns the similarity
         between artists i and j

        :param i: artist i int
        :param j: artist j int
        :return: similarity between i and j float
        """
        try:
            return self.similarities_map_artists[i][j]
        except KeyError:
            try:
                return self.similarities_map_artists[j][i]
            except KeyError:
                return 0.0
        except AttributeError:
            similarities = helper.read("artist-artist-similarities_bis" + str(self.test), ",")
            self.similarities_map_artists = {}
            old_x = 0
            for (x, y, value) in similarities:
                x = int(x)
                y = int(y)
                value = float(value)
                if old_x != x:
                    limit = 200
                    self.similarities_map_artists[x] = {}
                    self.similarities_map_artists[x][y] = value
                    old_x = x
                elif limit > 0:
                    self.similarities_map_artists[x][y] = value
                    limit -= 1

            try:
                return self.similarities_map_artists[i][j]
            except KeyError:
                try:
                    return self.similarities_map_artists[j][i]
                except KeyError:
                    return 0.0

    def get_user_similarities(self, i, j):
        """
        This method parse the item similairities csv and returns the similarity
         between artists i and j

        :param i: artist i int
        :param j: artist j int
        :return: similarity between i and j float
        """
        try:
            return self.similarities_map_users[i][j]
        except KeyError:
            try:
                return self.similarities_map_users[j][i]
            except KeyError:
                return 0.0
        except AttributeError:
            similarities = helper.read("user-user-similarities" + str(self.test), ",")
            self.similarities_map_users = {}
            old_x = 0
            for (x, y, value) in similarities:
                x = int(x)
                y = int(y)
                value = float(value)
                if old_x != x:
                    limit = 200
                    self.similarities_map_users[x] = {}
                    self.similarities_map_users[x][y] = value
                    old_x = x
                elif limit > 0:
                    self.similarities_map_users[x][y] = value
                    limit -= 1

            try:
                return self.similarities_map_users[i][j]
            except KeyError:
                try:
                    return self.similarities_map_users[j][i]
                except KeyError:
                    return 0.0

    def get_item_similarities_united_alt(self, i, j, type):
        """
        This method parse the item similairities csv and returns the similarity
         between tracks i and j

        :param i: track i int
        :param j: track j int
        :return: similarity between i and j float
        """
        while(True):
            try:
                value = self.similarities_map[i][j]
                for name, sim in value:
                    if name == type:
                        return sim
                return 0.0
            except KeyError:
                try:
                    value = self.similarities_map[j][i]
                    for name, sim in value:
                        if name == type:
                            return sim
                    return 0.0
                except KeyError:
                    return 0.0
            except AttributeError:
                similarities1 = helper.read("item-item-similarities" + str(self.test) + "_bis", ",")
                similarities2 = helper.read("item-item-similarities" + str(self.test), ",")
                similarities3 = helper.read("item-item-content" + str(self.test), ",")
                self.similarities_map = {}
                old_x = 0
                for (x, y, value) in similarities1:
                    x = int(x)
                    y = int(y)
                    value = ["users", float(value)]
                    if old_x != x:
                        limit = 150
                        self.similarities_map[x] = {}
                        self.similarities_map[x][y] = []
                        self.similarities_map[x][y].append(value)
                        old_x = x
                    elif limit > 0:
                        self.similarities_map[x][y] = []
                        self.similarities_map[x][y].append(value)
                        limit -= 1

                for (x, y, value) in similarities2:
                    x = int(x)
                    y = int(y)
                    value = ["playlists", float(value)]
                    if old_x != x:
                        limit = 150
                        try:
                            self.similarities_map[x][y].append(value)
                        except KeyError:
                            try:
                                self.similarities_map[x][y] = []
                                self.similarities_map[x][y].append(value)

                            except KeyError:
                                self.similarities_map[x] = {}
                                self.similarities_map[x][y] = []
                                self.similarities_map[x][y].append(value)

                        old_x = x
                    elif limit > 0:
                        try:
                            self.similarities_map[x][y].append(value)
                        except KeyError:
                            self.similarities_map[x][y] = []
                            self.similarities_map[x][y].append(value)
                        limit -= 1

                for (x, y, value) in similarities3:
                    x = int(x)
                    y = int(y)
                    value = ["content", float(value)]
                    if old_x != x:
                        limit = 150
                        try:
                            self.similarities_map[x][y].append(value)
                        except KeyError:
                            try:
                                self.similarities_map[x][y] = []
                                self.similarities_map[x][y].append(value)

                            except KeyError:
                                self.similarities_map[x] = {}
                                self.similarities_map[x][y] = []
                                self.similarities_map[x][y].append(value)
                        old_x = x
                    elif limit > 0:
                        try:
                            self.similarities_map[x][y].append(value)
                        except KeyError:
                            self.similarities_map[x][y] = []
                            self.similarities_map[x][y].append(value)
                        limit -= 1



    def set_item_similarities(self, i, j, update):
        """
        """

        try:
            value = self.similarities_map[(i,j)]
            self.similarities_map[(i,j)] = value + update
        except KeyError:
            try:
                value = self.similarities_map[(j,i)]
                self.similarities_map[(j,i)] = value + update
            except KeyError:
                pass

    def set_item_similarities_epoch(self, i, j, update):
        """
        """

        if i < j:
            value = self.similarities_map[i][j]
            self.similarities_map[i][j] = value + update
        elif i > j:
            value = self.similarities_map[j][i]
            self.similarities_map[j][i] = value + update

    def null_item_similarities(self, i, j):
        """
        """
        try:
            value = self.similarities_map[(i,j)]
            self.similarities_map[(i,j)] = 0
        except KeyError:
            try:
                value = self.similarities_map[(j,i)]
                self.similarities_map[(j,i)] = 0
            except KeyError:
                pass

    def null_item_similarities_alt(self, i, j):
        """
        """
        try:
            value = self.similarities_map[i][j]
            self.similarities_map[i][j] = 0
        except KeyError:
            try:
                value = self.similarities_map[j][i]
                self.similarities_map[j][i] = 0
            except KeyError:
                pass

    def normalized_tag_preference(self, playlist, active_tag=None):
        """
        """
        playlist_tracks = self.get_playlist_tracks(playlist)
        playlist_tags = [tag for track in playlist_tracks for tag in self.get_track_tags(track)]

        if active_tag:

            count_tags = playlist_tags.count(active_tag)
            normalized = float(count_tags) / (1 + len(playlist_tags))

            return normalized

        else:
            tags_counter = Counter(playlist_tags)
            mean_average = helper.mean(tags_counter.values())

            return mean_average




    def get_playlist_significative_tag(self, playlist):
        """
        Try to catch the most significative tag in the playlist

        :param playlist: playlist
        :return tag: int tag
        """
        tags = [tag for track in self.get_playlist_tracks(playlist) for tag in self.get_track_tags(track)]
        tags_counter = Counter(tags).items()
        mean = helper.mean([value for tag, value in tags_counter])
        try:
            max_value = max(tags_counter, key=itemgetter(1))
        except ValueError:
            return -1
        if max_value[1] > mean:
            return max_value[0]
        else:
            return -1

    def get_track_significative_titles(self, track):
        """
        Try to guess what's the most significant titles for the track if any

        :param track: track
        :return: titles (int list)
        """
        titles = self.get_track_titles(track)
        counter_titles = Counter(titles)
        mean = helper.mean([value for title, value in counter_titles.iteritems()])
        return [title for title in titles if counter_titles[title] > mean]

    def get_train_list(self):
        """
        getter for the train_list

        :return: train list
        """

        return self.train_list

    def get_playlist_relevant_tracks(self, active_playlist):
        """
        Getter for the test methods which return the tracks of the given playlist which are in the test set

        :param active_playlist: the playlist for which return the tracks
        :return: a set of tracks
        """
        test_set = self.get_test_set()
        return [track for playlist, track in test_set if active_playlist == playlist]

    def load_train_list(self, test=None):
        """
        Loader for train list when the engine start
        If the test mode is off load train final
        else load the corresponding train set

        :param test: specify the istance of test if enabled
        :return: None
        """
        if self.track_whitelist:
            try:
                fp = open('data/track_whitelist', 'rb')
                track_whitelist = set(helper.parseList(fp.readline(), int))
                fp.close()
                logging.debug("Loaded track whitelist!")
            except:
                self.track_whitelist = False
                logging.debug("No track whitelist file found, continuing with all tags!")

        if test == None:
            to_open = "train_final"
            train_list = helper.read(to_open)
            logging.debug("loaded %s" % (to_open))
            train_list.next()
            if self.track_whitelist:
                self.train_list = [[int(element[0]), int(element[1])] for element in train_list if int(element[1]) in track_whitelist]
            else:
                self.train_list = [[int(element[0]), int(element[1])] for element in train_list]
            self.num_interactions = len(self.train_list)
        else:

            to_open = "train_set"+str(test)
            train_list = helper.read(to_open)
            logging.debug("loaded %s" % (to_open))
            if self.track_whitelist:
                self.train_list = [[int(element[0]), int(element[1])] for element in train_list if int(element[1]) in track_whitelist]
            else:
                self.train_list = [[int(element[0]), int(element[1])] for element in train_list]
            self.num_interactions = len(self.train_list)

    def get_tag_playlists_map(self):
        """
        getter for the hashmap which associate tag with the playlist that included tracks with such tag

        [tag] -> [playlist0, playlist1, ...]

        :return: the hashmap
        """
        try:
            return self.tag_playlists_map
        except AttributeError:
            self.favourite_user_track_map = {}
            self.tag_playlists_map = defaultdict(lambda: [], {})
            for playlist in self.get_playlists():
                tracks = self.get_playlist_tracks(playlist)  # get already included tracks
                tags = [tag for track in tracks for tag in self.get_track_tags(track)]
                tags_set = set(tags)
                for tag in tags_set:
                    self.tag_playlists_map[tag].append(playlist)
            # mad things -> self.tag_playlists_map = {key: Counter(value) for key, value in self.tag_playlists_map.iteritems()}
            return self.tag_playlists_map


    def genetic(self, active_tag):
        """
        Get the value defined by the gene in case of DEAP working

        :param active_tag: tag to be keep or discarded
        :return: 1 if tag is in the gene 0 otherwise
        """
        while True:
            if self.individual:
                try:
                    return self.encoding[active_tag]
                except AttributeError:
                    fp = open("data/tag_encoding", "rb")
                    self.tag_encoding = helper.parseList(fp.readline(), int)
                    if len(self.individual_coefficient) != len(self.tag_encoding):
                        raise ValueError("individual - tag association mismatch! Please check tag encoding")
                        sys.exit(0)
                    self.encoding = {tag: individual for individual, tag in zip(self.individual_coefficient, self.tag_encoding)}
                except KeyError:
                    return 1.0
            else:
                return 1.0

    def get_track_users(self, track):
        """
        track -> [users]

        :param: track
        :return: int list
        """
        playlists = self.get_track_playlists(track)
        playlist_final = self.get_playlist_final()
        while True:
            try:
                return self.track_users_map[track]
            except AttributeError:
                self.track_users_map = {}
            except KeyError:
                users = [playlist_final[playlist][4] for playlist in playlists]
                users_set = set(users)
                self.track_users_map[track] = users_set

    def get_playlist_tracks_ratings(self, playlist):
        """
        Get the ratings of the tracks in the playlist where each rating is the number of times the track
        has been included by the user

        :param playlist: the user playlist
        :return: a list of rating respecting the track position
        """
        user_tracks = self.get_playlist_user_tracks(playlist)
        tracks = self.get_playlist_tracks(playlist)
        return [user_tracks.count(track) for track in tracks]

    def get_user_rating(self, playlist, track):
        """
        """
        user_tracks = self.get_playlist_user_tracks(playlist)
        return user_tracks.count(track)

    def get_track_inclusion_value(self, track):
        """
        Get the actual number of playlist which included the track

        :return: integer number
        """
        track_playlists_map = self.get_track_playlists_map()
        list_playlists = track_playlists_map[track]
        return len(list_playlists)

    def get_average_global_track_inclusion(self):
        """
        Getter for the global average inclusion coefficient of a track
        :return: a float coefficient
        """
        try:
            return self.avg_global_track
        except AttributeError:

            track_playlists_map = self.get_track_playlists_map()
            list_playlists = track_playlists_map.values()
            list_lenght = [len(playlists) for playlists in list_playlists]
            self.avg_global_track = sum(list_lenght)/float(len(list_lenght))
            return self.avg_global_track

    def get_title_playlists_map(self):
        """
        Getter for the hashmap which associate a specific title with all the playlist that have such title
        [title] -> [playlist1, playlist2, ...]

        :return: an hashmap
        """

        try:
            return self.title_playlists_map
        except AttributeError:
            self.title_playlists_map = defaultdict(lambda: [], {})
            playlist_final = self.get_playlist_final()
            for playlist in playlist_final:
                titles = playlist_final[playlist][1]  # get already included titles
                for title in titles:
                    self.title_playlists_map[title].append(playlist)
            return self.title_playlists_map

    def get_max_inclusion_value(self):
        """
        Getter for the maximum inclusion for a single track

        :return: an integer coefficient
        """
        try:
            return self.max_inclusion_value
        except AttributeError:
            self.max_inclusion_value = max([len(items) for items in self.get_tag_playlists_map().values()])
            return self.max_inclusion_value

    def get_num_tag(self):
        """
        Getter for the total number of tags currently loaded

        :return: number of loaded tags
        """
        try:
            return self.num_tag
        except AttributeError:
            tracks_map = self.get_tracks_map()
            self.num_tag = len(set([tag for values in tracks_map.itervalues() for tag in values[4]]))
            return self.num_tag

    def get_track_idf(self, track):
        """
        getter for the idf of the track (BM-25 version)

        :param track: the track
        :result: the float idf
        """
        n = len(self.get_track_playlists(track))
        N = len(self.get_tracks_map())
        idf = math.log((N - n + 0.5) / (n + 0.5), 10)
        return idf

    def get_tag_tracks(self, tag):
        """
        getter for tracks which contain the tag
        """

        try:
            return self.tag_tracks[tag]
        except AttributeError:
            self.tag_tracks = defaultdict(lambda: [])
            tracks_map = self.get_tracks_map()
            for track in tracks_map:
                tags = tracks_map[track][4]
                for feat in tags:
                    self.tag_tracks[feat].append(track)
            return self.tag_tracks[tag]

    def get_tag_playlists(self, tag):
        """
        Getter for the playlists which include the given tag

        :param tag: tag to which playlists need to be taken
        :return: list of playlists
        """
        try:
            return self.tag_playlists_map[tag]
        except AttributeError:
            tag_playlist_map = self.get_tag_playlists_map()
            playlist_tags_included = tag_playlist_map[tag]
            return playlist_tags_included

    def get_num_playlists(self):
        """
        Getter for the actual number of playlists in the database

        :return: integer number of playlists
        """
        try:
            return self.num_playlists
        except AttributeError:
            self.num_playlists = len(self.get_playlists())
            return self.num_playlists

    def get_tag_idf(self, tag, tf_idf):
        """
        returns the idf of a specific tag BM-25

        :param tag: tag for which the idf is needed
        :return: idf (float)
        """
        while True:
            try:
                return self.idf_map[tag]
            except AttributeError:
                self.idf_map = {}
            except KeyError:
                n = float(len(self.get_tag_playlists(tag)))
                N = self.get_num_playlists()
                if tf_idf == "bm25":
                    idf = 0.5 + math.log10((N - n + 0.5) / (n + 0.5))
                elif tf_idf == "normal":
                    idf = math.log1p(N/n)
                self.idf_map[tag]=idf
                return idf

    def get_target_score(self, playlist, track, k=1.2, b=0.75):
        """
        score as defined by bm-25 calculated on global basis

        :param playlist: playlist(document)
        :param track: query for the document
        :param k: as defined by bm-25
        :param b: as defined by bm-25
        :return: float score
        """
        documents_number = len(self.get_playlists())
        tag_playlist_map = self.get_tag_playlists_map()
        average = self.get_average_playlist_tags_count()
        playlist_features = [tag for item in self.get_playlist_tracks(playlist) for tag in self.get_track_tags(item)]
        document_len = len(playlist_features)
        if document_len == 0:
            raise ValueError

        scores = []
        for tag in self.get_track_tags(track):
            document_containing_tag = len(tag_playlist_map[tag])
            tag_count = playlist_features.count(tag)
            tf = (tag_count * (k+1)) / (tag_count + (k * (1 - b + b * (document_len / average))))
            idf = math.log10((documents_number - document_containing_tag + 0.5)/(document_containing_tag + 0.5))
            scores.append(tf*idf)

        score = sum(scores)
        return score

    def get_tag_idf_track(self, tag):
        """
        :param N:
        :param n:
        :return: idf
        """
        try:
            return self.tag_idf_track_map[tag]
        except AttributeError:
            self.tag_idf_track_map = {}
            track_map = self.get_tracks_map()
            self.tracks_number = len(self.get_tracks())
            self.track_tags_map_a = [set(lst[4]) for lst in track_map.values()]
            tags = self.track_tags_map_a
            N = self.tracks_number
            n = sum([1.0 for lst in tags if tag in lst])
            idf = math.log(1.0 + (N/n), 10)
            self.tag_idf_track_map[tag] = idf
            return idf
        except KeyError:
            tags = self.track_tags_map_a
            N = self.tracks_number
            n = sum([1.0 for lst in tags if tag in lst])
            idf = math.log(1.0 + (N/n), 10)
            self.tag_idf_track_map[tag] = idf
            return idf


    def get_playlist_final(self):
        """
        the initialization of the playlist final csv
        :return:
        """

        try:
            return self.playlist_final
        except AttributeError:
            playlist_list = helper.read("playlists_final")
            result = {}
            playlist_list.next()
            if self.title_whitelist:
                try:
                    fp = open('data/title_whitelist', 'rb')
                    title_whitelist = set(helper.parseList(fp.readline(), int))
                    fp.close()
                    logging.debug("Loaded title whitelist!")
                except:
                    self.title_whitelist = False
                    logging.debug("No title whitelist file found, continuing with all tags!")
            for playlist in playlist_list:
                created_at = int(playlist[0])
                playlist_id = int(playlist[1])
                titles = helper.parseList(playlist[2], int)
                if self.title_whitelist:
                    titles = [title for title in titles if title in title_whitelist]
                numtracks = int(playlist[3])
                duration = int(playlist[4])
                owner = int(playlist[5])

                result[playlist_id]= [created_at, titles, numtracks, duration, owner]
            self.playlist_final = result
            return self.playlist_final

    def get_playlist_numtracks(self, playlist):
        """
        Get the number of tracks of the given playlist as specified in playlist_final.csv

        :param playlist: int playlist
        :return: int numtracks
        """
        return self.playlist_final[playlist][2]

    def user_user_similarity(self, active_user, knn=75):
        """
        Get similarity value based on the user owner of the playlist
        NOTE: NOT RECOMMENDED as abandoned

        :param active_user:
        :param knn:
        :return: similarities list
        """
        similarities = []
        active_user_tracks = set(self.get_user_tracks(active_user))
        for user in self.get_user_set():
            user_tracks = set(self.get_user_tracks(user))
            try:

                coefficient = len(active_user_tracks.intersection(user_tracks)) / (float(len(active_user_tracks.union(user_tracks))))

            except ZeroDivisionError:
                continue
            similarities.append([user, coefficient])
        similarities.sort(key=itemgetter(1), reverse=True)
        return similarities[0:knn]

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
        target_playlists = helper.read("target_playlists")
        target_playlists.next()
        self.target_playlists = [int(elem[0]) for elem in target_playlists]

    def get_target_tracks(self, filter_duration=True):
        """
        getter for the tracks to be recommended
        if the target_tracks does not exists it create the list from the corresponding csv
        :return:
        """
        try:
            return self.target_tracks
        except:
            # the set cast is just for commodity
            target_tracks = helper.read("target_tracks")
            target_tracks.next()
            self.target_tracks = (int(track[0]) for track in target_tracks)
            if filter_duration:
                self.target_tracks = [track for track in self.target_tracks if self.get_track_duration(track) <= 0 or self.get_track_duration(track) > 30000]
            self.target_tracks = set(self.target_tracks)
            return self.target_tracks

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

    def get_tags(self):
        """
        """
        try:
            return self.tags
        except AttributeError:
            self.tags = list(set([tag for item in self.get_tracks_map().itervalues() for tag in item[4]]))
            return self.tags

    def get_users(self):
        """
        """
        playlist_final = self.get_playlist_final()
        users = [playlist[4] for playlist in playlist_final.values()]
        set_users = set(users)
        return set_users

    def get_average_playlist_tags_count(self):
        """
        get the average tag count of playlists of the dataset
        :return: average
        """
        try:
            return self.average_playlist_length
        except AttributeError:
            playlist_tracks_map = self.get_playlist_tracks_map()
            tracks_list = playlist_tracks_map.values()
            ravanello = [len([tag for track in tracks for tag in self.get_track_tags(track)]) for tracks in tracks_list]
            self.average_playlist_length =  helper.mean(ravanello)
            return self.average_playlist_length

    def get_average_track_tags_count(self):
        """
        get the average tag count per track of the dataset
        """
        try:
            return self.average_tags_length
        except AttributeError:
            track_map = self.get_tracks_map()
            tags_list_len = [len(lst[4]) for lst in track_map.itervalues()]
            self.average_tags_length = helper.mean(tags_list_len)
            return self.average_tags_length

    def get_created_at_tracks(self, track):
        """
        This method returns for a track, the created at attributes of the playlists where the specific track was included
        """

        playlists_track = self.get_track_playlists(track)

        return [self.get_created_at_playlist(playlist) for playlist in playlists_track]




    def compute_tracks_map(self):
        """
        parse tracks_final.csv dividing all field into the corresponding part of the list
        :return:
        """
        tracks = helper.read("tracks_final")
        tracks.next()
        result = {}
        if self.tag_whitelist:
            try:
                fp = open('data/tag_whitelist', 'rb')
                tag_whitelist = set(helper.parseList(fp.readline(), int))
                fp.close()
                logging.debug("Loaded tag whitelist!")
            except:
                self.tag_whitelist = False
                logging.debug("No tag whitelist file found, continuing with all tags!")

        for track in tracks:
            track_id = int(track[0])
            artist_id = int(track[1]) + 276615
            duration = int(track[2])
            try:
                playcount = float(track[3]) # yes, PLAYCOUNT is memorized as a floating point
            except ValueError:
                playcount = 0.0
            album = helper.parseList(track[4], int)
            try:
                album = int(album[0]) + 847203  # yes again, album is memorized as a list, even if no track have more than 1 album
            except:
                album = -1
            tags = helper.parseList(track[5], int) # evaluation of the tags list
            if self.extended:
                tags = [artist_id] + [album] + tags
                tags = [tag for tag in tags if tag > 0]
            if self.tag_whitelist:
                tags = [tag for tag in tags if tag in tag_whitelist]
            try:
                if self.individual:
                    tags = [tag for tag in tags if math.ceil(self.genetic(tag))]
            except Exception as e:
                print e
            result[track_id]= [artist_id, duration, playcount, album, tags]

        return result

    def get_playlist_user_playlists(self, playlist):
        """
        return the playlists of the user who created the given playlist
        :param playlist: the playlist of the user
        :return: a list of playlists
        """
        playlist_final = self.get_playlist_final()
        owned_by = playlist_final[playlist][4]

        owner_playlist = self.get_owner_playlists()
        playlist_list = owner_playlist[owned_by]

        return playlist_list

    def get_playlist_user_tracks(self, playlist):
        """
        return the tracks of the user who created the given playlist
        :param playlist: the playlist of the user
        :return: a list of tracks
        """
        playlist_final = self.get_playlist_final()
        owned_by = playlist_final[playlist][4]

        owner_playlist = self.get_owner_playlists()
        playlist_list = owner_playlist[owned_by]

        tracks_listened = [track for playlist in playlist_list for track in self.get_playlist_tracks(playlist)]

        return tracks_listened

    def get_user_playlists(self, user):
        """
        return the playlists of the user
        :param user: the user
        :return: a list of playlists
        """
        owner_playlist = self.get_owner_playlists()
        return owner_playlist[user]


    def get_playlist_user(self, playlist):
        """
        Getter for the owner of the playlist

        :param playlist: playlist
        :return: owner(user)
        """
        playlist_final = self.get_playlist_final()
        return playlist_final[playlist][4]


    def get_owner_playlists(self):
        """
        map owner: playlists

        :return: hashmap
        """
        try:
            return self.owner_playlist

        except AttributeError:
            playlists = self.get_playlist_final()
            self.owner_playlist = defaultdict(lambda: [], {})
            for playlist in playlists:
                owned_by= playlists[playlist][4]
                self.owner_playlist[owned_by].append(playlist)

            return self.owner_playlist

    def get_created_at_playlist(self, playlist):
        """
        Get the created_at timestamp of a given playlist

        :param playlist: playlist
        :return: timestamp
        """
        playlists = self.get_playlist_final()

        return playlists[playlist][0]


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

    def get_tracks(self):
        """
        gettter for the tracks in train, so the item in URM

        :return: a set of tracks
        """
        try:
            return self.tracks_set
        except AttributeError:
            train = self.get_train_list()
            self.tracks_set = set([track for playlist, track in train])
            return self.tracks_set

    def get_num_tracks(self):
        """
        getter for the total number of tracks in the URM

        :return: number of tracks
        """
        try:
            return self.num_tracks
        except AttributeError:
            self.num_tracks = len(self.get_tracks())
            return self.num_tracks

    def get_track_tags(self, track):
        """
        compute tags from tracks_final given a specific track

        :param track:
        :return:
        """
        while True:
            try:
                tags = self.tracks_map[track][4]
                return tags
            except AttributeError:
                track_map = self.get_tracks_map()
            except LookupError:
                return []

    def get_track_tags_map(self):
        """
        compute hashmap to store track -> [tag1, tag2, ...] for fast retrieval
        :return: hashmap
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

        :param playlist: playlist for we need tracks
        :return: list of tracks
        """
        try:
            return self.playlist_tracks_map[target_playlist]
        except AttributeError:
            self.playlist_tracks_map = defaultdict(lambda: [], {})
            train_list = self.get_train_list()
            [self.playlist_tracks_map[playlist].append(track) for playlist, track in train_list]
            return self.playlist_tracks_map[target_playlist]

    def get_playlist_tracks_map(self):
        """
        Getter for the playlist_tracks map
        :return: hashmap
        """
        return self.playlist_tracks_map

    def get_track_playlists(self, track, n=None):
        """
        return all the playlists which includes the specified track
        :param track:
        :return:
        """
        try:
            if n:
                return [playlist for playlist in self.track_playlists_map[track] if len(self.get_playlist_tracks(playlist)) >= n]
            else:
                return self.track_playlists_map[track]
        except AttributeError:
            track_playlists_map = self.get_track_playlists_map()
            return track_playlists_map[track]

    def get_track_titles(self, track):
        """
        Get all the titles associated to the track
        track -> playlists -> titles

        :param track: track
        :return: list of titles
        """
        while True:
            try:
                return self.track_titles_map[track]
            except AttributeError:
                self.track_titles_map = {}
            except KeyError:
                playlists = self.get_track_playlists(track, n=10)
                titles_track = [title for playlist in playlists for title in self.get_playlist_titles(playlist)]
                self.track_titles_map[track] = titles_track

    def get_titles(self):
        """
        Getter for all the titles in the database

        :return: list of titles (int)
        """
        playlist_final = self.get_playlist_final()
        return set([title for playlist in playlist_final for title in playlist_final[playlist][1]])

    def get_playlist_titles(self, playlist):
        """
        Getter for the playlist titles

        :param playlist: the playlist
        :return: list of titles
        """
        try:
            return self.playlists_final[playlist][1]
        except AttributeError:
            playlist_final = self.get_playlist_final()
            titles = playlist_final[playlist][1]
            return titles

    def get_title_idf(self, title):
        """
        gets the smooth idf of a specific title
        :return:
        """
        title_playlist_map = self.get_title_playlists_map()
        playlist_including_title = title_playlist_map[title]
        den_idf = float(len(playlist_including_title))
        num_idf = self.get_num_playlists()
        idf = math.log1p(num_idf / den_idf) #smooth logarithm ln(1+x)
        return idf

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

    def get_top_included(self):
        """
        Return a list of tuples (track, value) where value is the number of featured playlists

        :return: top included list
        """
        try:
            return self.top_included
        except AttributeError:
            track_playlists_map = self.get_track_playlists_map()
            track_playlists = track_playlists_map.items()
            self.top_included = sorted(map(lambda x: [x[0], len(x[1])], track_playlists), key=lambda x: x[1], reverse=True)
            return self.top_included

    def get_track_count_included(self, track):
        """
        gets the number of playlists where the given track is included
        :return:
        """
        track_playlists_map = self.get_track_playlists_map()
        return len(track_playlists_map[track])

    def get_track_playcount(self, track):
        """
        gets the playcount of the given track
        :param track:
        :return:
        """
        map_tracks = self.get_tracks_map()
        return map_tracks[track][2]

    def get_artist(self, track):
        """
        Getter for the artist who made the track

        :param track: A single track
        :return: the artist id integer
        """
        map_tracks = self.get_tracks_map()
        return map_tracks[track][0]

    def get_album(self, track):
        """
        Getter for the album who made the track

        :param track: A single track
        :return: the album id integer
        """
        map_tracks = self.get_tracks_map()
        return map_tracks[track][3]

    def get_artist_tracks(self, artist):
        """
        Getter for all the songs of the artist

        :param artist: An artist
        :return: the songs list
        """
        while True:
            try:
                return self.artist_tracks_map[artist]
            except AttributeError:
                self.artist_tracks_map = defaultdict(lambda: [], {})
                map_tracks = self.get_tracks_map()
                for track in map_tracks:
                    artist_a = map_tracks[track][0]
                    self.artist_tracks_map[artist_a].append(track)

    def get_track_album_tracks(self, track):
        """
        Getter of all the songs of the album which contains also the current track
        :param track: A single track
        :return: the songs list
        """
        map_tracks = self.get_tracks_map()
        album = map_tracks[track][3]
        return [track for track in map_tracks if album == map_tracks[track][3]]



    def get_top_listened(self):
        """
        return a list of tuples (track, value) where
        value is the number of times played

        :return: the top listened list sorted
        """
        try:
            return self.top_listened
        except AttributeError:
            tracks = self.get_tracks_map()
            self.top_listened = [(track, tracks[track][2]) for track in tracks]
            self.top_listened.sort(key=itemgetter(1), reverse=False)
            return self.top_listened
