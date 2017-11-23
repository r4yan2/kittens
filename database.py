import math
from collections import defaultdict, Counter
import helper
import random
import logging
from operator import itemgetter
import sqlite3

class Database:
    def __init__(self, test):
        """
        if we are in test execution train_set, test_set are loaded transparently,
        otherwise the normal dataset are loaded

        :param test: istance
        :return: the initialized object
        """
        connection = sqlite3.connect("/dev/shm/db")
        connection.row_factory = lambda cursor, row: row[0]
        cursor = connection.cursor()
        self.cursor = cursor
        self.test = test

        if test > 0:
            self.load_test_set(test)
            self.load_train_list(test)
        else:
            self.load_train_list()

    def shrink_and_clean_db(self, knn=150):
        """
        Invoke some pragmas to shrink, clean and reoder the database
        """
        tracks = self.get_tracks()
        for i in tracks:
            keep = set(self.cursor.execute("select j from similarities_epoch where i=(?) ordered by value desc limit (?)", (i, knn)).fetchall())
            remove = tracks.difference(keep)
            self.cursor.execute('delete from similarities_epoch where i=%i and j in (%s)' % (i, ", ".join([str(track) for track in remove])))
                
        self.cursor.execute("vacuum")
        self.cursor.execute("PRAGMA optimize")
        self.cursor.execute("PRAGMA shrink_memory")

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
   
    def get_prediction_from_similarities_epoch(self, i, playlist_tracks):
        """
        """
        values = self.cursor.execute('select value from similarities_epoch where i=%i and j in (%s)' % (i, ", ".join([str(track) for track in playlist_tracks]))).fetchall()
        return sum(values)

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
        getter for the playlists set

        :return: set of playlists
        """
        try:
            return self.playlists
        except AttributeError:
            playlists = self.get_train_list()
            self.playlists = list(set([playlist for playlist, track in playlists]))
            return self.playlists

    def compute_test_set(self):
        """
        Computing the test set if Testing mode is enabled:

        * REMOVE a number of lines from the train set and put them in the test set
        * every line is randomly chosen to avoid over fitting of the test set

        :return: None
        """
        train = self.get_train_list()
        playlists_length = 5000
        tracks_length = 16097
        self.test_set = []
        playlists_with_n_tracks = set([playlist for playlist in self.get_playlists() if len(self.get_playlist_tracks(playlist)) >= 10])
        already_selected = set()
        already_selected_playlists = set()
        already_selected_tracks = set()
        while True:
            line = random.randint(0, len(train) - 1)
            playlist = train[line][0]
            track = train[line][1]
            if line not in already_selected:
                if len(already_selected_playlists) < playlists_length and len(already_selected_tracks) < tracks_length:
                    self.test_set.append(train[line])
                    already_selected_playlists.add(playlist)
                    already_selected_tracks.add(track)
                elif len(already_selected_playlists) >= playlists_length and len(already_selected_tracks) < tracks_length and playlist in already_selected_playlists:
                    self.test_set.append(train[line])
                    already_selected_tracks.add(track)
                elif len(already_selected_playlists) < playlists_length and len(already_selected_tracks) >= tracks_length and track in already_selected_tracks:
                    self.test_set.append(train[line])
                    already_selected_playlists.add(playlist)

            if len(already_selected_playlists) >= playlists_length and len(already_selected_tracks) >= tracks_length:
                break
        self.train_list = helper.diff_test_set(train, self.test_set)

    def get_user_set(self):
        """
        Getter for the user set taken from the pre-built urm

        :return: user set
        """
        try:
            return self.user_set
        except AttributeError:
            self.user_set = set([int(u) for (u,i,r) in helper.read("urm", ',')])
            return self.user_set

    def get_user_tracks(self, user):
        """
        Getter for the tracks of the user, taken from the pre-built urm

        :param user: the user for which we need the tracks
        :return: the list of tracks
        """
        try:
            return self.user_tracks[user]
        except AttributeError:
            self.user_tracks = defaultdict(lambda: [], {})
            for (u,i,r) in helper.read("urm", ','):
                self.user_tracks[int(u)].append(int(i))
            return self.user_tracks[user]

    def compute_content_playlists_similarity(self, playlist_a, knn=75, title_flag=1, tag_flag=0, track_flag=0, coefficient="jaccard"):
        """
        This method compute the neighborhood for a given playlists by using content or collaborative techniques

        :param playlist_a: given playlist_a
        :param knn: cardinality of the neighborhood
        :param title_flag: specify if title have to be used
        :param tag_flag: specify if tag have to be used
        :param track_flag: specify if track have to be used
        :coefficient: specify which coefficient to use
        :return: neighborhood
        """
        playlists = self.get_playlists()
        playlist_a_tags = []
        playlist_a_titles = []
        playlist_a_tracks = []
        tf_idf_playlist_a_tag = []
        tf_idf_playlist_a_title = []
        tf_idf_playlist_a_track = []


        if tag_flag:
            playlist_a_tags = list(set([tag for track in self.get_playlist_tracks(playlist_a) for tag in self.get_track_tags(track)]))

        if title_flag:
            playlist_a_titles = self.get_titles_playlist(playlist_a)
            playlist_a_titles_set = set(playlist_a_titles)

        if track_flag:
            playlist_a_tracks = self.get_playlist_tracks(playlist_a)

        if not (len(playlist_a_tags) and tag_flag or len(playlist_a_titles) and title_flag or len(playlist_a_tracks) and track_flag):
            raise ValueError("cannot generate neighborhood for", playlist_a)

        if tag_flag and coefficient == "cosine":
            tf_idf_playlist_a_tag = [(1.0 + math.log(playlist_a_tags.count(tag),10)) * self.get_tag_idf(tag) for tag in playlist_a_tags]

        if title_flag and coefficient == "cosine":
            tf_idf_playlist_a_title = [self.get_title_idf(title) for title in playlist_a_titles]

        if track_flag and coefficient == "cosine":
            tf_idf_playlist_a_track = [(1.0 + math.log(playlist_a_tracks.count(track),10)) * self.get_track_idf(track) for track in playlist_a_tracks]

        if coefficient == "cosine":
            tf_idf_playlist_a = tf_idf_playlist_a_tag + tf_idf_playlist_a_title + tf_idf_playlist_a_track

        neighborhood = []

        for playlist_b in playlists:

            similarity = []
            playlist_b_tags = []
            playlist_b_titles = []
            playlist_b_tracks = []
            tf_idf_playlist_b_tag = []
            tf_idf_playlist_b_title = []
            tf_idf_playlist_b_track = []

            if tag_flag:
                playlist_b_tags = list(set([tag for track in self.get_playlist_tracks(playlist_b) for tag in self.get_track_tags(track)]))

            if title_flag:
                playlist_b_titles = self.get_titles_playlist(playlist_b)

            if track_flag:
                playlist_b_tracks = self.get_playlist_tracks(playlist_b)

            if not (len(playlist_b_tags) and tag_flag or len(playlist_b_titles) and title_flag or len(playlist_b_tracks) and track_flag):
                continue

            if tag_flag and coefficient == "cosine":
                tf_idf_playlist_b_tag = [(1.0 + math.log(playlist_b_tags.count(tag),10)) * self.get_tag_idf(tag) for tag in playlist_b_tags]

            if title_flag and coefficient == "cosine":
                tf_idf_playlist_b_title = [self.get_title_idf(title) for title in playlist_b_titles]

            if track_flag and coefficient == "cosine":
                tf_idf_playlist_b_track = [self.get_track_idf(track) for track in playlist_b_tracks]

            tf_idf_playlist_b = tf_idf_playlist_b_tag + tf_idf_playlist_b_title + tf_idf_playlist_b_track

            num_cosine_sim_tag = 0
            num_cosine_sim_title = 0
            num_cosine_sim_track = 0

            if tag_flag and coefficient == "cosine":
                num_cosine_sim_tag = sum([tf_idf_playlist_a_tag[playlist_a_tags.index(tag)] * tf_idf_playlist_b_tag[playlist_b_tags.index(tag)] for tag in playlist_b_tags if tag in playlist_a_tags])

            if title_flag and coefficient == "cosine":
                num_cosine_sim_title = sum([tf_idf_playlist_a_title[playlist_a_titles.index(title)] * tf_idf_playlist_b_title[playlist_b_titles.index(title)] for title in playlist_b_titles if title in playlist_a_titles])
            elif title_flag and coefficient == "jaccard":
                try:
                    similarity = ( len(playlist_a_titles_set.intersection(playlist_b_titles))/float(len(playlist_a_titles_set.union(playlist_b_titles))) ) * (1.0 - (len(playlist_a_titles_set.union(playlist_b_titles).difference(playlist_a_titles_set.intersection(playlist_b_titles)))/float(len(playlist_a_titles_set.union(playlist_b_titles)))))
                except:
                    continue
            if track_flag and coefficient == "cosine":
                num_cosine_sim_track = sum([tf_idf_playlist_a_track[playlist_a_tracks.index(track)] * tf_idf_playlist_b_track[playlist_b_tracks.index(track)] for track in playlist_b_tracks if track in playlist_a_tracks])

            if coefficient == "cosine":
                num_cosine_sim = num_cosine_sim_tag + num_cosine_sim_title + num_cosine_sim_track
                den_cosine_sim = math.sqrt(sum([i ** 2 for i in tf_idf_playlist_a])) * math.sqrt(sum([i ** 2 for i in tf_idf_playlist_b]))

                try:
                    similarity = num_cosine_sim / den_cosine_sim
                except ZeroDivisionError:
                    continue
            neighborhood.append([playlist_b, similarity])
        knn_neighborg = [playlist for playlist, value in sorted(neighborhood[0:knn], key=itemgetter(1), reverse=True)]
        return knn_neighborg

    def get_min_max_playlists(self):
        """
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
        """
        try:
            return self.min_track, self.max_track
        except AttributeError:
            tracks = self.get_tracks()
            self.min_track = min(tracks)
            self.max_track = max(tracks)
            return self.min_track, self.max_track

    def compute_collaborative_playlists_similarity(self, playlist, knn=50, tracks_knn=None, coefficient="jaccard", values="None"):
        """
        This method computes the similarities between playlists based on the included tracks.
        Various coefficient can be used (jaccard, map, cosine, pearson)
        NB: the best is jaccard

        :param playlist: active_playlist
        :param knn: cardinality of the neighborhood
        :param coefficient: coefficient to use
        :param values: if None return only the playlists, if all return also the values
        :return: list of playlists
        """
        playlists = self.get_playlists()
        tracks_playlist = set(self.get_playlist_tracks(playlist))
        tracks_playlist_length = len(tracks_playlist)
        created_at_active = self.get_created_at_playlist(playlist)
        similarities = []

        for playlist_b in playlists:

            created_at = self.get_created_at_playlist(playlist_b)
            if not math.fabs(created_at_active - created_at) < (60 * 60 * 24 * 365 * 3):
                continue

            tracks_playlist_b = set(self.get_playlist_tracks(playlist_b))
            tracks_playlist_b_length = len(tracks_playlist_b)

            matched = tracks_playlist.intersection(tracks_playlist_b)
            matched_len = len(matched)
            not_matched = tracks_playlist.union(tracks_playlist_b).difference(matched)
            not_matched_len = len(not_matched)

            MSE = not_matched_len / float(len(tracks_playlist.union(tracks_playlist_b)))

            if coefficient == "jaccard":
                numerator = matched_len
                denominator = len(tracks_playlist.union(tracks_playlist_b))
                jaccard = numerator / float(denominator)
                MSD = 1.0 - (len(tracks_playlist.union(tracks_playlist_b).difference(tracks_playlist.intersection(tracks_playlist_b)))/float(len(tracks_playlist.union(tracks_playlist_b))))
                similarities.append([playlist_b, jaccard * MSD])

            elif coefficient == "map":
                # MAP@k it may be useful if only we know how to use it
                tag_mask = [float(track in tracks_playlist) for track in tracks_playlist_b]
                p_to_k_num = helper.multiply_lists(tag_mask, helper.cumulative_sum(tag_mask))
                p_to_k_den = range(1,len(tag_mask)+1)
                p_to_k = helper.divide_lists(p_to_k_num, p_to_k_den)
                try:
                    map_score = sum(p_to_k) / len(tracks_playlist)
                except ZeroDivisionError:
                    continue
                similarities.append([playlist_b,map_score])

            elif coefficient == "cosine":
                num_cosine_sim = matched_len

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
            tracks = []
            iterator = 0
            while len(tracks) < tracks_knn:
                tracks += self.get_playlist_tracks(similarities[iterator][0])
                iterator += 1
            return tracks

    def get_user_based_collaborative_filtering(self, active_playlist, knn=20, coefficient="jaccard"):
        """
        This method is created in order to provide a CF via users.
        Calculate the most K similar users to the active one by jaccard index

        :param playlist: active playlist
        :param knn: cardinality of the neighborhood
        :param coefficient: coefficient to use
        :return: list of tracks of the knn similar users
        """

        active_tracks = self.get_playlist_user_tracks(active_playlist)
        active_tracks_counter = Counter(active_tracks)
        active_tracks_set = set(active_tracks)
        already_scanned_user = set()
        playlists = self.get_playlists()
        created_at_active = self.get_created_at_playlist(active_playlist)

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
            tracks_set = set(tracks)

            if coefficient == "wtf":
                # Jaccard
                numerator = sum([active_tracks_counter[track] * tracks_counter[track] for track in tracks if track in active_tracks])
                denominator = sum([elem * elem for elem in active_tracks_counter.values()]) \
                              + sum([elem * elem for elem in tracks_counter.values()]) - numerator
                try:
                    jaccard = numerator / float(denominator)
                except ZeroDivisionError:
                    continue

                mean_square_difference = 1.0 - sum([(active_tracks_counter[track] - tracks_counter[track]) ** 2 for track in active_tracks if track in tracks])

                similarity = jaccard * mean_square_difference

            elif coefficient == "jaccard":
                similarity = helper.jaccard(active_tracks_set, tracks_set)

            neighborhood.append([tracks, similarity])
        neighborhood.sort(key=itemgetter(1), reverse=True)
        return [track for tracks, value in neighborhood[0:knn] for track in tracks]

    def get_knn_track_similarities(self, active_track, knn=50):
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
            self.get_item_similarities(0,0)
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
            similarities = helper.read("item-item-similarities1", ",")
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
        """
        return self.num_interactions

    def init_item_similarities_epoch(self):
        """
        Init the similarities map used later for the epoch iteration method
        """
        self.cursor.execute("drop table if exists similarities_epoch")
        self.cursor.execute("CREATE TABLE 'similarities_epoch' ('i' INTEGER, 'j' INTEGER, 'value' FLOAT, PRIMARY KEY(i,j))")

    def get_item_similarities_epoch(self, i, j):
        """
        """
        if i != j:
            try:
                return self.cursor.execute("select value from similarities_epoch where i=(?) and j=(?)", (i,j)).next()
            except:
                return random.random()
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
            similarities = helper.read("item-item-similarities1", ",")
            self.similarities_map = {}
            old_x = 0
            for (x, y, value) in similarities:
                x = int(x)
                y = int(y)
                value = float(value)
                if old_x != x:
                    limit = 399
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

        if i != j:
            self.cursor.execute("INSERT OR REPLACE INTO similarities_epoch (i, j, value) VALUES ((?), (?), (?))", (i, j , update))

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
        if self.test == 1:
            return self.cursor.execute("select track_id from test_set1 where  playlist_id = (?) ", (active_playlist,)).fetchall()

    def load_train_list(self, test=None):
        """
        Loader for train list when the engine start
        If the test mode is off load train final
        else load the corresponding train set

        :param test: specify the istance of test if enabled
        :return: None
        """
        if test == None:
            train_list = helper.read("train_final")
            train_list.next()
            self.train_list = [[int(element[0]), int(element[1])] for element in train_list]
            self.num_interactions = len(self.train_list)
        else:
            train_list = helper.read("train_set"+str(test))
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
                tags = [tag for track in tracks for tag in self.get_track_tags(track)]# + [item * -1 * (10 ** 10) for track in tracks for item in self.get_favourite_user_track(track)]
                #tags = [tag for track in tracks for tag in self.get_track_tags(track)] + [item * -1 * (10**10) for item in self.get_titles_track(track)]
                tags_set = set(tags)
                for tag in tags_set:
                    self.tag_playlists_map[tag].append(playlist)
            return self.tag_playlists_map

    def get_favourite_user_track(self, track):
        """
        Currently I don't know what this method do
        track -> user?

        :return:
        """
        playlists = self.get_track_playlists(track)
        playlist_final = self.get_playlist_final()
        try:
            return self.favourite_user_track_map[track]
        except KeyError:

            users = [playlist_final[playlist][4] for playlist in playlists]
            users_set = set(users)
            if users == []:
                self.favourite_user_track_map[track] = []
            else:
                max_count = max([users.count(user) for user in users_set])
                self.favourite_user_track_map[track] = [user for user in users_set if users.count(user) == max_count]
            return self.favourite_user_track_map[track]

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

    def get_average_track_inclusion(self, track):
        """
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
            for playlist in self.get_playlists():
                titles = self.get_titles_playlist(playlist)  # get already included titles
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

    def get_track_idf(self, track):
        """
        getter for the idf of the track

        :param track: the track
        :result: the float idf
        """
        n = len(self.get_track_playlists(track))
        N = len(self.get_tracks_map())
        idf = math.log((N - n + 0.5) / (n + 0.5), 10)
        return idf

    def get_tag_tracks(self, tag):
        """
        getter for the idf of the tag with respect to the
        """

        try:
            return self.tags_list[tag]
        except AttributeError:
            self.tags_list = {}
            tracks_map = self.get_tracks_map()
            tags_list = [tag for items in tracks_map.values() for tag in items[4]]
            tags_list_set = set(tags_list)
            for tag in tags_list_set:
                self.tags_list[tag] = tags_list.count(tag)
            return self.tags_list[tag]

    def get_tag_idf(self, tag):
        """
        returns the idf of a specific tag with respect to the playlists which includes such tag
        :return: idf
        """
        tag_playlist_map = self.get_tag_playlists_map()
        playlist_tags_included = tag_playlist_map[tag]
        N = len(self.get_playlists())
        n = float(len(playlist_tags_included))
        try:
            idf = 0.5 + math.log((N - n + 0.5) / (n + 0.5), 10)
        except ValueError:
            idf = 0
        except ZeroDivisionError:
            idf = 0
        return idf

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
            for playlist in playlist_list:
                created_at = int(playlist[0])
                playlist_id = int(playlist[1])
                title = helper.parseIntList(playlist[2])
                numtracks = int(playlist[3])
                duration = int(playlist[4])
                owner = int(playlist[5])

                result[playlist_id]= [created_at, title, numtracks, duration, owner]
            self.playlist_final = result
            return self.playlist_final

    def user_user_similarity(self, active_user, knn=75):
        """
        """
        similarities = []
        active_user_tracks = set(self.get_user_tracks(active_user))
        for user in self.get_user_set():
            user_tracks = set(self.get_user_tracks(user))
            try:

                coefficient = len(active_user_tracks.intersection(user_tracks)) / (float(len(active_user_tracks.union(user_tracks))) \
                                                                                   - len(active_user_tracks.intersection(user_tracks)))
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
        if self.test == 1:
            return self.cursor.execute("select distinct playlist_id from test_set1").fetchall()
        return self.cursor.execute("select distinct playlist_id from target_playlists").fetchall()


    def compute_target_playlists(self):
        """
        return the list from the csv of the target_playlist, with the first row removed
        :return:
        """
        target_playlists = helper.read("target_playlists")
        target_playlists.next()
        self.target_playlists = [int(elem[0]) for elem in target_playlists]

    def get_target_tracks(self):
        """
        getter for the tracks to be recommended
        if the target_tracks does not exists it create the list from the corresponding csv
        :return:
        """
        if self.test == 1:
            return self.cursor.execute("select distinct track_id from test_set1").fetchall()
        return self.cursor.execute("select distinct track_id from target_tracks").fetchall()

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


    def compute_tracks_map(self):
        """
        parse tracks_final.csv dividing all field into the corresponding part of the list
        :return:
        """
        tracks = helper.read("tracks_final")
        tracks.next()
        result = {}
        iterator = -10
        for track in tracks:
            track_id = int(track[0])
            artist_id = int(track[1])
            duration = int(track[2])
            try:
                playcount = float(track[3]) # yes, PLAYCOUNT is memorized as a floating point
            except ValueError:
                playcount = 0.0
            album = helper.parseIntList(track[4])
            try:
                album = int(album[0])  # yes again, album is memorized as a list, even if no track have more than 1 album
            except:
                album = iterator
                iterator -= 1
            tags = helper.parseIntList(track[5]) # evaluation of the tags list

            tags_extended = [artist_id + 276615] + [album + 847203 if album > 0 else iterator] + [playcount + 1064529] + tags

            result[track_id]= [artist_id, duration, playcount, album, tags_extended]
        return result

    def get_playlist_user_tracks(self, playlist):
        """
        return the tracks of the user who created the given playlist
        :param playlist: the playlist of the user
        :return: a list of tracks
        """
        return self.cursor.execute("select track_id from train_set1 \
                                    where playlist_id = ( select playlist_id from playlists_final \
                                    where owner = ( select owner from playlists_final \
                                    where playlist_id = (?)))", (playlist,)).fetchall()

    def get_user_playlists(self, playlist, user=None):
        """
        return the playlists of the user who created the given playlist
        :param playlist: the playlist of the user
        :return: a list of tracks
        """
        owner_playlist = self.get_owner_playlists()
        if user:
            return owner_playlist[user]
        playlist_final = self.get_playlist_final()
        owned_by = playlist_final[playlist][4]

        playlist_list = owner_playlist[owned_by]

        return playlist_list

    def get_playlist_user(self, playlist):
        """

        :return:
        """
        return self.cursor.execute("select owner from playlists_final where playlist_id = (?)", (playlist,)).fetchall()

    def get_owner_playlists(self):
        """

        :return:
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

        :return:
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
        track_map = self.get_tracks_map()
        try:
            return track_map[track][4]
        except LookupError:
            return []

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
        return self.cursor.execute("select track_id from train_set1 where playlist_id = (?)", (target_playlist,)).fetchall()


    def get_playlist_tracks_map(self):
        """

        """
        return self.playlist_tracks_map

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


    def get_titles_track(self, track):
        """


        :return:
        """
        playlists = self.get_track_playlists(track)
        titles_track = [title for playlist in playlists for title in self.get_titles_playlist(playlist)]

        return titles_track

    def get_titles_playlist(self, playlist):
        """
        Getter for the playlist titles

        :param playlist: the playlist
        :return: list of titles
        """
        #TODO SISTEMARE TUTTO
        playlist_final = self.get_playlist_final()
        titles = playlist_final[playlist][1]
        return titles

    def get_title_idf(self, title):
        """
        gets the idf of a specific title
        :return:
        """
        title_playlist_map = self.get_title_playlists_map()
        playlist_titles_included = title_playlist_map[title]
        den_idf = float(len(playlist_titles_included))
        num_idf = len(self.get_playlists())
        try:
            idf = math.log(1.0 + (num_idf / den_idf), 10)
        except ValueError:
            idf = 0
        except ZeroDivisionError:
            idf = 0
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

    def get_artist_tracks(self, track):
        """
        Getter for all the songs of the artist who made the track

        :param track: A single track
        :return: the songs list
        """
        map_tracks = self.get_tracks_map()
        artist = map_tracks[track][0]
        return [track for track in map_tracks if artist == map_tracks[track][0]]

    def get_album_tracks(self, track):
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
            tracks = self.get_tracks()
            self.top_listened = sorted(map(lambda x: [x[0], x[3]], tracks), key=lambda x: x[1], reverse=True)
            return self.top_listened
