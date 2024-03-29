# modules
import csv
from collections import defaultdict
import math
import sys
import time
import random
from operator import itemgetter
import logging
import helper
from collections import Counter
from database import Database

class Recommender:
    def __init__(self, db=None):
        """
        Init method for recommender class. The parameters may be always empty, except for debug cases in which
        one may want to pass a database istance to use

        :param db: Defaulted to None, only for terminal debugging
        :return: the allocated object
        """
        if db:
            self.db = db

    def check_recommendations(self, playlist, recommendations):
        """
        Test the selected recommendation method with different metrics:
        * Map@5
        * Precision
        * Recall

        :param playlist: playlist to be tested
        :recommendations: recommendations to be tested
        :raise ValueError: in case the recommendations provided have more/less then 5 items
        :return: the triple with the test results
        """
        test_set = self.db.get_playlist_relevant_tracks(playlist)
        test_set_length = len(test_set)
        if len(recommendations) != 5:
            raise ValueError('Recommendations list have less than 5 recommendations')

        is_relevant = [float(item in test_set) for item in recommendations]
        is_relevant_length = len(is_relevant)

        # MAP@5
        p_to_k_num = helper.multiply_lists(is_relevant, helper.cumulative_sum(is_relevant))
        p_to_k_den = range(1,is_relevant_length+1)
        p_to_k = helper.divide_lists(p_to_k_num, p_to_k_den)
        try:
            map_score = sum(p_to_k) / min(test_set_length, is_relevant_length)
        except ZeroDivisionError:
            map_score = 0

        if map_score > 1.0001: #.0001 avoid matching artifact due to float operations
            raise ValueError("Detected anomalous MAP@5, please investigate!")

        # Precision
        try:
            precision = sum(is_relevant)/is_relevant_length
        except ZeroDivisionError:
            precision = 0

        # Recall
        try:
            recall = sum(is_relevant)/test_set_length
        except ZeroDivisionError:
            recall = 0

        ranks = range(is_relevant_length)
        pos_ranks = [pos for pos in ranks if is_relevant[pos]]
        pos_ranks_length = len(pos_ranks)
        neg_ranks = [pos for pos in ranks if not is_relevant[pos]]
        neg_ranks_length = len(neg_ranks)
        auc_score = 0.0
        if neg_ranks_length == 0:
            auc_score = 1.0
        elif pos_ranks_length > 0:
            for pos_pred in pos_ranks:
                auc_score += sum([1 for elem in neg_ranks if pos_pred < elem])
            auc_score /= (pos_ranks_length * neg_ranks_length)

        # return the quartet
        return map_score, auc_score, precision, recall

    def deactivate_individual(self):
        """
        """
        if self.db.get_individual_state():
            self.db.set_individual_state(False)


    def activate_individual(self):
        """
        """
        if not self.db.get_individual_state():
            try:
                self.db.set_individual_state(True)
            except Exception as e:
                logging.debug("%s" % (e))


    def run(self, choice, db, q_in, q_out, test, number):
        """
        Main loop for the recommender worker, it fetch a new playlist from the general queue until it get a -1, then terminate

        :param choice: The choice of recommendation method to uses
        :param db: The database instance
        :param q_in: The input queue from which playlist are fetched
        :param q_out: The output queue in which store results when ready
        :param test: Flag which indicate if this is a test istance
        :param number: Worker number identifier
        :return: None
        """

        # Retrieve the db from the list of arguments
        self.db = db
        self.test = test

        if choice == 20:
            # do some epoch pre-processing on data
            for i in xrange(1,101):
                logging.debug("epoch %i/100" % i)
                self.epoch_iteration()
            choice = 11

        # main loop for the worker
        while True:

            # getting the data from the queue in
            (identifier, target) = q_in.get()

            # if is the end stop working
            if target == -1:
                break
            # else start the hard work
            logging.debug("worker %i took playlist %i" % (number, target))
            recommendations = []

            if choice == 0:
                recommendations = self.make_random_recommendations(target)

            elif choice == 1:
                recommendations = self.make_top_listened_recommendations(target)

            elif choice == 2:
                recommendations = self.make_top_included_recommendations(target)

            elif choice == 3:
                try:
                    recommendations = self.make_top_tag_recommendations(target)
                except ValueError as e: # No features
                    logging.debug("cannot use top tag for %i because of %s" % (target, e))
                    #q_out.put(-1)

            elif choice == 4:
                try:
                    recommendations = self.make_tf_idf_recommendations(target)
                except ValueError as e: # No tracks or features
                    logging.debug("cannot use tf_idf for %i because of %s" % (target, e))
                    #q_out.put(-1)

            elif choice == 5:
                try:
                    recommendations = self.combined_top_tag_tfidf_recommendations(target)
                except ValueError:
                    recommendations = self.make_top_included_recommendations(target)

            elif choice == 6:
                try:
                    recommendations = self.ensemble_recommendations(target)
                except ValueError as e: # No tracks or features
                    logging.debug("ValueError by playlist %i, %s" % (target, e))

            elif choice == 7:
                try:
                    recommendations = self.make_tf_idf_titles_recommendations(target)
                except ValueError as e:
                    logging.debug("titles recommendations cannot be done for %i because of %s" % (target, e))
                    try:
                        recommendations = self.make_tf_idf_recommendations(target)
                    except ValueError:
                        recommendations = self.make_top_included_recommendations(target)

            elif choice == 8:
                try:
                    recommendations = self.combined_collaborative_top_tag(target)
                except ValueError as e:
                    logging.debug("cannot use top tag for %i because of %s" % (target, e))
                    #q_out.put(-1)

            elif choice == 9:
                recommendations = self.combined_top_tag_tfidf_titles_recommendations(target)

            elif choice == 10:
                recommendations = self.combined_tfidf_tfidf_titles_recommendations(target)

            elif choice == 11:
                try:
                    recommendations = self.make_collaborative_item_item_recommendations(target)
                except ValueError as e:
                    logging.debug("cannot use collaborative for %i because of %s" % (target, e))
                    #q_out.put(-1)

            elif choice == 12:
                try:
                    recommendations = self.make_artists_recommendations(target)
                except LookupError:
                    # TODO implement user-targeted recommendations
                    recommendations = self.make_top_included_recommendations(target)
                except ValueError: # no dominant artist
                    try:
                        recommendations = self.make_tf_idf_recommendations(target)
                    except ValueError: # this may happen when the playlist have 1-2 tracks with no features (fuck it)
                        recommendations = self.make_top_included_recommendations(target)
                if len(recommendations) < 5: # if there are not enough artist tracks to recommend or if the tracks have a strage avg duration
                    recommendations = self.make_some_padding(target, recommendations)

            elif choice == 13:
                try:
                    recommendations = self.make_hybrid_recommendations(target)
                except Exception as e:
                    logging.debug("hybrid method exception, need to padd %s" % (e))
            elif choice == 14:
                try:
                    recommendations = self.make_neighborhood_similarity(target)
                except ValueError: # No tracks or features
                    recommendations = self.make_top_included_recommendations(target)
                if len(recommendations) < 5: # padding needed
                    recommendations = self.make_some_padding(target, recommendations)

            elif choice == 15:
                try:
                    recommendations = self.make_user_based_recommendations(target)
                except Exception as e:
                    logging.debug("cannot use user based for %i because of %s" % (target, e))
                    #q_out.put(-1)

            elif choice == 16:
                try:
                    recommendations = self.make_naive_bayes_recommendations(target)
                except ValueError as e:
                    logging.debug("cannot use naive bayes for %i because of %s" % (target, e))
                    #q_out.put(-1)

            elif choice == 17:
                try:
                    recommendations = self.make_bad_tf_idf_recommendations(target)
                except ValueError as e:
                    logging.debug("cannot use bad tf_idf for %i because of %s" % (target, e))
                    #q_out.put(-1)

            elif choice == 18:
                try:
                    recommendations = self.make_bad_tf_idf_recommendations_jaccard(target)
                except ValueError as e:
                    logging.debug("cannot use bad tf_idf for %i because of %s" % (target, e))
                    #q_out.put(-1)

            elif choice == 19:
                try:
                    recommendations = self.make_playlist_based_recommendations(target)
                except ValueError as e:
                    logging.debug("cannot use playlist based for %i because of %s" % (target, e))
                    #q_out.put(-1)
            elif choice == 21:
                try:
                    recommendations = self.tf_idf_bm25_recommendations(target)
                except ValueError as e:
                    pass

            padding = True
            if len(recommendations) < 5:
                logging.debug("padding needed")
                if padding:
                    recommendations = self.make_some_padding(target, recommendations=recommendations)
                else:
                    continue

            # doing testing things if test mode enabled
            if test:
                map5, auc_score, precision, recall = self.check_recommendations(target, recommendations)
                q_out.put([map5, auc_score, precision, recall, number, target, len(db.get_playlist_tracks(target))])

            else:
                # else put the result into the out queue
                q_out.put([identifier, target, recommendations])

    def make_some_padding(self, playlist, recommendations):
        """
        Make some padding when needed

        :param playlist: target playlist
        :param recommendations: actual recommended tracks
        :return: the full list of tracks recommended to the target playlist
        """
        methods = [4, 3, 2] # equal to choice parameter
        i = 0
        while len(recommendations) < 5:
            if methods[i] == 4:
                try:
                    recommendations = self.make_tf_idf_recommendations(playlist, recommendations=recommendations, coefficient="cosine", tf_idf="bm25")
                except ValueError:
                    pass
            if methods[i] == 3:
                try:
                    recommendations = self.make_top_tag_recommendations(playlist, recommendations=recommendations)
                except ValueError:
                    pass
            elif methods[i] == 2:
                recommendations = self.make_top_included_recommendations(playlist, recommendations=recommendations)
            i = (i + 1) % len(methods)

        logging.debug("Padding for playlist: %i generated prediction: %s" % (playlist, recommendations))

        return recommendations

    def make_random_recommendations(self, playlist, target_tracks=[], recommendations=[], knn=5):
        """
        Take 5 random tracks and recommend them

        :param playlist: Target playlist
        :param target_tracks: Set of target in which choose the random one
        :param recommendations: Set of recommendations already included
        :param knn: Number of items to recommend
        :return: Recommendations
        """
        if target_tracks == []:
            target_tracks = self.db.get_target_tracks()
        knn -= len(recommendations)
        count = 0
        already_included = self.db.get_playlist_tracks(playlist)
        num_target_tracks = len(target_tracks)
        while count < knn:
            track = random.randint(0, num_target_tracks)
            if track not in already_included and track not in recommendations:
                recommendations.append(target_tracks[track])
                count += 1
        return recommendations

    def make_top_listened_recommendations(self, playlist, target_tracks=[], recommendations=[], knn=5, reverse_order=True):
        """
        Recommend the knn top listened tracks in the dataset

        :param playlist: Target playlist
        :param target_tracks: Set of target in which choose the random one
        :param recommendations: Set of recommendations already included
        :param knn: Number of items to recommend
        :return: Recommendations
        """
        if target_tracks == []:
            target_tracks = self.db.get_target_tracks()
            already_included = self.db.get_playlist_tracks(playlist)
            target_tracks = target_tracks.difference(already_included).difference(recommendations)
        knn -= len(recommendations)
        predictions = [[track, self.db.get_track_playcount(track)] for track in target_tracks]
        recs = sorted(predictions, key=itemgetter(1), reverse=reverse_order)[:knn]
        logging.debug("recommendations: %s" % (recs))

        return recommendations + [recommendation for recommendation, value in recs]

    def make_top_included_recommendations(self, playlist, target_tracks=[], recommendations=[], knn=5):
        """
        Recommend the most playlists-included tracks

        :param playlist: Target playlist
        :param target_tracks: Set of target in which choose the random one
        :param recommendations: Set of recommendations already included
        :param knn: Number of items to recommend
        :return: Recommendations
        """
        if target_tracks == []:
            target_tracks = self.db.get_target_tracks()

        knn -= len(recommendations)
        top_included = self.db.get_top_included()
        iterator = 0
        count = 0
        already_included = self.db.get_playlist_tracks(playlist)
        while count < knn:
            item = top_included[iterator][0]
            if item not in already_included and item not in recommendations and item in target_tracks:
                recommendations.append(item)
                count += 1
            iterator += 1

        return recommendations

    def make_top_tag_recommendations(self, active_playlist, target_tracks=[], recommendations=[], knn=5, ensemble=0, neighborhood_knn=0):
        """
        This method takes into account tags. For the active playlist all tags of the tracks are computed,
        then for every recommendable track the comparison of the tags is used taking into account:

        * the matched tags with respect to the total tags number of the playlists (precision)
        * secondly the matched tags over the total tags of the track (recall)
        * lastly the position of the track in top included

        :param active_playlist: Target playlist
        :param target_tracks: Set of target in which choose the random one
        :param recommendations: Set of recommendations already included
        :param knn: Number of items to recommend
        :return: Recommendations
        """
        active_tracks = self.db.get_playlist_tracks(active_playlist) # get already included tracks

        tot_tags = self.db.get_num_tag()
        if neighborhood_knn:
            if target_tracks != []:
                raise ValueError("target track conflict with neighborhood")

            target_tracks = self.db.compute_collaborative_playlists_similarity(active_playlist, tracks_knn=neighborhood_knn)
        else:
            if target_tracks == []:
                target_tracks = self.db.get_target_tracks()

        knn -= len(recommendations)
        already_included = active_tracks
        active_tags = [tag for track in active_tracks for tag in self.db.get_track_tags(track)]
        active_tags_set = set(active_tags)
        active_tags_unique = list(active_tags_set)

        if active_tags == []:
            raise ValueError("target playlist empty")

        top_tracks = []

        for track in target_tracks: # make the actual recommendation

            if track in recommendations:
                continue
            tags = self.db.get_track_tags(track)
            '''
            try:
                # calculate first parameter: matched over playlist tags set
                matched = active_tags_set.intersection(tags) # calculate the tags which match
                norm_playlist = math.sqrt(len(active_tags_set))
                norm_track = math.sqrt(len(tags))
                value_a = len(matched)/float(norm_playlist*norm_track)
            except ZeroDivisionError:
                value_a = 0
            except ValueError:
                value_a = 0
            '''
            jaccard = helper.jaccard(tags, active_tags_set)
            #value_b = sum([self.db.get_item_similarities_alt(track,j) for j in active_tracks])

            #tot_tags = self.db.get_num_tag()

            #value_phi = helper.phi_coefficient(active_tags_unique, tags, tot_tags)

            top_tracks.append([track, jaccard])

        top_tracks.sort(key=itemgetter(1), reverse=True)
        if ensemble:
            return top_tracks[0:knn]
        logging.debug("recommendations for playlist %i:%s" % (active_playlist, top_tracks[:5]))
        return recommendations + [recommendation[0] for recommendation in top_tracks[0:knn]]

    def combined_collaborative_top_tag(self, active_playlist):
        """
        """
        filtered_tracks = self.make_collaborative_item_item_recommendations(active_playlist, knn=75)
        return self.make_top_tag_recommendations(active_playlist, target_tracks=filtered_tracks)

    def tf_idf_bm25_recommendations(self, active_playlist):
        score = []
        for track in self.db.get_target_tracks():
            score.append([track, self.db.get_target_score(active_playlist, track)])
        score = sorted(score, key=itemgetter(1), reverse=True)
        recommendations = [item for item, value in score[0:5]]
        return recommendations

    def make_tf_idf_recommendations(self, active_playlist, target_tracks=[], neighborhood_knn=0, recommendations=[], knn=5, ensemble=0, tf_idf="bm25", coefficient="cosine", shrink=True):
        """
        Make Recommendations based on the tf-idf of the track tags

        :param active_playlist: Target playlist
        :param target_tracks: Set of target in which choose the random one
        :param recommendations: Set of recommendations already included
        :param neighborhood_knn: neighborhoood to get to filter target tracks
        :param knn: Number of items to recommend
        :param ensemble: Specify if the output of the method will be used in an ensemble
        :param tf_idf: Specify with tf-idf method to used
        :param coefficient: Specify how to calculate the final score
        :return: Recommendations
        :raise ValueError: In case playlist have no tracks or tracks with no features
        """

        playlist_tracks = self.db.get_playlist_tracks(active_playlist)
        #playlist_track_playlists = [self.db.get_track_playlists(t) for t in playlist_tracks]
        playlist_tracks_set = set(playlist_tracks)
        #target_playlist_tags = [tag for track in playlist_tracks for tag in self.db.get_track_tags(track)]
        if playlist_tracks == []:
            raise ValueError("playlist is empty")

        if target_tracks == []:
            target_tracks = self.db.get_target_tracks().difference(recommendations).difference(playlist_tracks)

            if neighborhood_knn:
                try:
                    neighborhood_tracks = self.db.compute_content_playlists_similarity(active_playlist, tracks_knn = [1,300])
                except Exception:
                    neighborhood_tracks = [{},{}]
                #if len(tracks) > 0:
                    #target_tracks = target_tracks.intersection(tracks)

        knn = knn - len(recommendations)

        possible_recommendations = []
        playlist_features = [tag for track in playlist_tracks for tag in self.db.get_track_tags(track)]
        playlist_features_set = set(playlist_features)
        playlist_features_unique = list(playlist_features_set)
        playlist_features_length = float(len(playlist_features))
        #playlist_titles = self.db.get_playlist_titles(active_playlist)
        #significative_tag = self.db.get_playlist_significative_tag(active_playlist)
        #playlist_duration = [self.db.get_track_duration(track) for track in playlist_tracks]
        #playlist_duration_avg = sum(playlist_duration) / len(playlist_tracks)
        playlist_artists = [self.db.get_artist(track) for track in playlist_tracks]

        #active_user_playlists = self.db.get_playlist_user_playlists(active_playlist)
        #active_tracks_user = [track for playlist in active_user_playlists for track in self.db.get_playlist_tracks(playlist)]
        #set_active_tracks_user = set(active_tracks_user)
        #active_artists = [self.db.get_artist(track) for track in active_tracks_user]
        #active_artists_filtered = [a for a in active_artists if a not in playlist_artists]
        #set_active_artists = set(active_artists)


        if len(playlist_features) == 0:
            raise ValueError("playlist have no features!")

        #target_tracks_artists = [track for track in target_tracks if self.db.get_artist(track) in set_active_artists]

        artists_percentages = []
        playlist_artists_counter = Counter(playlist_artists)

        most_in_artist = max(playlist_artists_counter.values())/float(sum(playlist_artists_counter.values()))

        if most_in_artist < 0.10:
            #similar_artists_tracks = [(t, self.db.get_track_inclusion_value(t) + v) for a in scanned for aa, v in self.db.get_similar_artist(a) for t in self.db.get_artist_tracks(aa) if t in target_tracks
                                        #and self.db.get_track_playcount(t) / (1.0 + self.db.get_track_inclusion_value(t)) >= 1]
            similar_artists_tracks = [t for a in playlist_artists_counter for aa, v in self.db.get_similar_artist(a) for t in self.db.get_artist_tracks(aa) if t in target_tracks
                                        and self.db.get_track_playcount(t) / (1.0 + self.db.get_track_inclusion_value(t)) >= 1]
            target_tracks = similar_artists_tracks

            self.deactivate_individual()

        else:
            self.activate_individual()

        if tf_idf == "bm25":
            average = self.db.get_average_playlist_tags_count()
            k = 1.2
            b = 0.75
            len_over_avg = k * (1 - b + b * playlist_features_length/average)
            tf_idf_playlist = [self.db.get_tag_idf(tag, tf_idf) * ((playlist_features.count(tag) * (k + 1)) / (playlist_features.count(tag) + len_over_avg)) for tag in playlist_features_unique]
        elif tf_idf == "normal":
            tf_idf_playlist = [(1.0 + math.log10(playlist_features.count(tag))) * self.db.get_tag_idf(tag, tf_idf) for tag in playlist_features_unique]

        average_track_tags_count = self.db.get_average_track_tags_count()

        for track in target_tracks:
            artist = self.db.get_artist(track)
            track_titles = set(self.db.get_track_titles(track))
            most_title = self.db.get_track_significative_titles(track)
            tags = self.db.get_track_tags(track)
            '''
            titles_count = 0
            for title in playlist_titles:
                if title == most_title:
                    titles_count += 1
            '''
            count = 0

            if tags == []:
                continue
            for tag in playlist_features_set:
                if tag in tags:
                    count += 1
            '''
            track_playlists = self.db.get_track_playlists(track)
            track_playlists_tracks = [t for playlist_t in track_playlists for t in self.db.get_playlist_tracks(playlist_t)]
            difference = len(playlist_tracks_set.difference(track_playlists_tracks))
            sum_distance = 0
            for playlist_t in track_playlists:
                tracks_p = self.db.get_playlist_tracks(playlist_t)

                sum_distance += sum([1 for t in playlist_tracks if t in tracks_p])

            val = float(sum_distance) / (sum_distance + difference)


            track_playlists = self.db.get_track_playlists(track)
            len_track_playlists = len(track_playlists)


            track_playlists_tracks = (self.db.get_playlist_tracks(playlist_t) for playlist_t in track_playlists)
            values = [0 for _ in xrange(len(playlist_tracks))]
            for tracks in track_playlists_tracks:
                for t in playlist_tracks:
                    if t in tracks:
                        values[playlist_tracks.index(t)] += 1


            values = 0
            for playlist_t in playlist_track_playlists:
                values += len([1 for elem in track_playlists if elem in playlist_t])

            try:
                confidence = float(values) / (len_track_playlists * len(playlist_tracks))
            except Exception:
                confidence = 1
            '''

            if tf_idf == "bm25":
                tf_track = (k + 1) / (1 + k * (1 - b + b * (len(tags) / average_track_tags_count))) # tags.count(tag) is always 1
                tf_idf_track = [self.db.get_tag_idf(tag, tf_idf) * tf_track for tag in tags]
            elif tf_idf == "normal":
                tf_idf_track = [self.db.get_tag_idf(tag, tf_idf) for tag in tags]

            if coefficient == "pearson":

                mean_tfidf_track = helper.mean(tf_idf_track)
                mean_tfidf_playlist = helper.mean(tf_idf_playlist)

                numerator = sum([(tf_idf_track[tags.index(tag)] - mean_tfidf_track) *
                               (tf_idf_playlist[playlist_features_unique.index(tag)] - mean_tfidf_playlist)
                               for tag in tags if tag in playlist_features_unique])

                denominator = math.sqrt(sum([(i - mean_tfidf_playlist) ** 2 for i in tf_idf_playlist]) *
                sum([(i - mean_tfidf_track) ** 2 for i in tf_idf_track]))

            elif coefficient == "cosine":

                hemming = helper.hemming_distance(tags, playlist_features_set)
                #jaccard_titles = helper.jaccard(playlist_titles, track_titles)
                jaccard = helper.jaccard(tags, playlist_features_set) * (1.0 / len(playlist_tracks))
                #phi = helper.phi_coefficient(tags, playlist_features, tot_tags)
                #tags_len = len(tags)
                #euclidean = helper.euclidean_squared_distance(tags, playlist_features_set)
                #playlist_features_len = len(playlist_features)
                value_d = sum([self.db.get_item_similarities_alt(track,j) for j in playlist_tracks])
                value_a = sum([self.db.get_artist_similarities(artist, j) for j in playlist_artists])
                #value_t = sum([self.db.get_tag_similarities(i,j) for i in tags for j in playlist_features_unique])
                ochiai = helper.ochiai_coefficient(tags, playlist_features)

                #track_duration = self.db.get_track_duration(track)

                numerator = helper.square_sum([tf_idf_track[tags.index(tag)] * self.db.genetic(tag) * tf_idf_playlist[playlist_features_unique.index(tag)] for tag in tags if tag in playlist_features_unique])

                denominator = math.sqrt(helper.square_sum(tf_idf_playlist)) * math.sqrt(helper.square_sum(tf_idf_track))
                if shrink:
                    denominator -= math.sqrt(self.db.get_track_inclusion_value(track))
                '''
                if sum(tf_idf_titles_playlist) > 0:

                    titles = self.db.get_track_titles(track)
                    titles_set = list(set(titles))

                    tf_idf_title = [(1.0 + math.log(titles.count(title), 10)) * self.db.get_title_idf(title) for title in titles_set]

                    num_cosine_sim = [tf_idf_title[titles_set.index(title)] * tf_idf_titles_playlist[playlist_titles.index(title)] for
                                      title in titles_set if title in playlist_titles]

                    den_cosine_sim = math.sqrt(sum([i ** 2 for i in tf_idf_titles_playlist])) * math.sqrt(
                        sum([i ** 2 for i in tf_idf_title]))
                    try:
                        cosine_sim = sum(num_cosine_sim) / den_cosine_sim
                    except ZeroDivisionError:
                        continue
                else:
                    cosine_sim = 0
                '''
            elif coefficient == "product":

                numerator1 = [tf_idf_track[tags.index(tag)] * tf_idf_playlist[playlist_features_unique.index(tag)] for tag in tags if tag in playlist_features_unique]
                mean_tfidf_track = sum(tf_idf_track) / len(tf_idf_track)
                mean_tfidf_playlist = sum(tf_idf_playlist) / len(tf_idf_playlist)

                numerator2 = [(tf_idf_track[tags.index(tag)] - mean_tfidf_track) *
                               (tf_idf_playlist[playlist_features_unique.index(tag)] - mean_tfidf_playlist)
                               for tag in tags if tag in playlist_features_unique]

                numerator = 0
                for a,b in zip(numerator1, numerator2):
                    numerator += a*b

                denominator1 = math.sqrt(helper.square_sum(tf_idf_playlist)) * math.sqrt(helper.square_sum(tf_idf_track))
                denominator2 = math.sqrt(sum([(i - mean_tfidf_playlist) ** 2 for i in tf_idf_playlist]) * sum([(i - mean_tfidf_track) ** 2 for i in tf_idf_track]))

                denominator = denominator1 * denominator2

                denominator += helper.set_difference_len(playlist_features_set, tags)
                if shrink:
                    denominator -= math.sqrt(self.db.get_track_inclusion_value(track))


            elif coefficient == "tanimoto":
                numerator = sum([tf_idf_track[tags.index(tag)] * tf_idf_playlist[playlist_features_unique.index(tag)] for tag in tags if tag in playlist_features_unique])

                denominator = sum(tf_idf_playlist) * sum(tf_idf_track) - numerator
                denominator += helper.set_difference_len(playlist_features_set, tags)

            if numerator and denominator:
                similarity = (value_a + 30 * jaccard + count * 0.75 + value_d * 25 + hemming / (0.25 * hemming + 1) + 0.75 * ochiai) * numerator / denominator

            else:
                continue

            #multiplier = 1.0
            '''
            if significative_tag in tags:
                multiplier += 0.15
            if track in neighborhood_tracks[1]:
                multiplier += 0.055
            if track in neighborhood_tracks[0]:
                multiplier += 0.30
            if track in artist_tracks:
                multiplier += 0.5

            if most_title in playlist_titles:
                multiplier += 0.5
            '''
            #multiplier += confidence

            #similarity *= multiplier

            if knn <= 5:
                if len(possible_recommendations) < 5:
                    possible_recommendations.append([track, similarity])
                    mini = min(possible_recommendations, key=itemgetter(1))
                else:
                    if similarity > mini[1]:
                        possible_recommendations.remove(mini)
                        possible_recommendations.append([track, similarity])
                        mini = min(possible_recommendations, key=itemgetter(1))
            else:
                possible_recommendations.append([track, similarity])

        if ensemble:
            return possible_recommendations[0:knn]
        else:
            possible_recommendations = sorted(possible_recommendations, key=itemgetter(1), reverse=True)
        recs = recommendations + [recommendation for recommendation, value in possible_recommendations]
        if knn <= 5:
            logging.debug("playlist: %i generated prediction: %s" % (active_playlist, possible_recommendations))
        #playcounts = [[track, self.db.get_track_playcount(track)] for track in recs]
        #sorted_playcounts = sorted(playcounts, key=itemgetter(1), reverse=False)
        #new_recs = [elem[0] for elem in sorted_playcounts]
        return recs

    def make_naive_bayes_recommendations(self, playlist, target_tracks=[], recommendations=[], knn=5, ensemble=0):
        """
        This method tries to implement a machine learning approach using statistic predictions for a specific track
        The main goal is, considering the tags of a track and that of a playlist, and computing the conditional probabilities
        between them, to esthimate how much a track fits for a specific playlist

        :param playlist: Target playlist
        :param recommendations: Set of recommendations already included
        :param knn: Number of items to recommend
        """

        target_tracks = set(self.db.get_target_tracks())

        possible_recommendations = []
        playlist_tracks = self.db.get_playlist_tracks(playlist)
        playlist_tags = [tag for track in playlist_tracks for tag in self.db.get_track_tags(track)]
        playlist_tags_counter = Counter(playlist_tags)
        len_playlist_tracks = float(len(playlist_tracks))
        playlist_tags_set = set(playlist_tags)
        len_playlist_tags_set = len(playlist_tags_set)
        likelihood_map = defaultdict(lambda: 0.0, {})

        p_in_playlist = len_playlist_tracks / self.db.get_num_tracks()
        playlist_tags_total = float(sum(playlist_tags_counter.values()))

        '''
        for tag in playlist_tags_set:
            p_tag_yes = playlist_tags_counter[tag] * self.db.get_tag_idf(tag) / float(sum(playlist_tags_counter.values()))
            p_yes = sum(playlist_tags_counter.values()) / float(len_playlist_tags_set)
            p_tag = playlist_tags_counter[tag] * self.db.get_tag_idf(tag) / float(len_playlist_tags_set)
            likelihood_map[tag] = (p_tag_yes * p_yes)
        '''
        for track in target_tracks:
            count = 0
            tags = self.db.get_track_tags(track)
            for tag in playlist_tags_set:
                if tag in tags:
                    count += 1

            # avg probability
            try:
                probability = helper.product([playlist_tags_counter[tag]/len_playlist_tracks * self.db.get_tag_idf(tag) for tag in tags]) * p_in_playlist / helper.product([playlist_tags_counter[tag]/ playlist_tags_total for tag in tags])
            except:
                continue

            possible_recommendations.append([track, count * probability])

        recs = sorted(possible_recommendations, key=itemgetter(1), reverse=True)
        if ensemble:
            return recs
        return recommendations + [recommendation for recommendation, value in recs[0:knn]]

    def make_knn_bayes_recommendations(self, playlist, recommendations=[], knn=5, ensemble=0):
        """
        This method tries to implement a machine learning approach using statistic predictions for a specific track
        The main goal is, considering the tags of a track and that of a playlist, and computing the conditional probabilities
        between them, to esthimate how much a track fits for a specific playlist

        :param playlist: Target playlist
        :param recommendations: Set of recommendations already included
        :param knn: Number of items to recommend
        """

        target_tracks = set(self.db.get_target_tracks())

        possible_recommendations = []
        playlist_tracks = self.db.get_playlist_tracks(playlist)
        playlist_tags = [tag for track in playlist_tracks for tag in self.db.get_track_tags(track)]
        playlist_tags_counter = Counter(playlist_tags)
        len_playlist_tracks = len(playlist_tracks)
        playlist_tags_set = set(playlist_tags)
        len_playlist_tags_set = len(playlist_tags_set)
        likelihood_map = defaultdict(lambda: 0.0, {})

        for tag in playlist_tags_set:
            p_tag_yes = playlist_tags_counter[tag] * self.db.get_tag_idf(tag) / float(sum(playlist_tags_counter.values()))
            p_yes = sum(playlist_tags_counter.values()) / float(len_playlist_tags_set)
            likelihood_map[tag] = (p_tag_yes * p_yes)

        for track in target_tracks:
            # bayesian probability non-optimized

            knn = 50
            neighborhood = self.db.get_item_similarities(track, knn)


            probability = prior_probability / likelihood

            possible_recommendations.append([track, probability])

        recs = sorted(possible_recommendations, key=itemgetter(1), reverse=True)[0:knn]
        if ensemble:
            return recs
        return recommendations + [recommendation for recommendation, value in recs]

    def make_hybrid_recommendations(self, active_playlist, recommendations=[], knn=5, ensemble=0):
        """
        Hybrid recommendations method which takes into account the number of tracks and titles in a playlist

        :param playlist: Target playlist
        :param recommendations: Actual recommendations made to the playlist
        :param knn: number of recommendations to generate
        :return: the new recommendations
        """
        playlist_tracks = self.db.get_playlist_tracks(active_playlist)
        len_playlist_tracks = len(playlist_tracks)
        playlist_tags = [tag for track in playlist_tracks for tag in self.db.get_track_tags(track)]
        len_playlist_tags = len(playlist_tags)
        playlist_titles = self.db.get_titles_playlist(active_playlist)
        len_playlist_titles = len(playlist_titles)
        active_playlist_creation = self.db.get_created_at_playlist(active_playlist)
        predictions = []
        '''
        # case: playlist have no tracks and no title
        if playlist_tracks == [] and playlist_titles == []:

            playlists = self.db.get_user_playlists(active_playlist)
            if playlists == [active_playlist]:
                recommendations = self.make_top_included_recommendations(active_playlist)
            owner_playlists = [playlist for playlist in playlists if math.fabs(self.db.get_created_at_playlist(playlist) - active_playlist_creation) < (60 * 60 * 24 * 180)]
            owner_playlists_tracks = [track for playlist in owner_playlists for track in self.db.get_playlist_tracks(playlist)]
            target_tracks = self.db.get_target_tracks()
            for track in target_tracks:
                prediction = sum([self.db.get_item_similarities_alt(track,j) for j in owner_playlists_tracks])
                predictions.append([track, prediction])
            predictions = sorted(predictions, key=itemgetter(1), reverse=True)
            if ensemble:
                return predictions[0:knn]
            return [recommendation for recommendation, _ in predictions[0:knn]]
        '''

        #if len_playlist_titles > 0:
            #recommendations = self.ensemble_recommendations(active_playlist, mask=[1, 0, 0, 0, 1, 0, 0, 0])

        if len_playlist_tracks >= 0 and len_playlist_tracks < 162:
            recommendations = self.make_tf_idf_recommendations(active_playlist)

        elif len_playlist_tracks >= 162:
            recommendations = self.make_bad_tf_idf_recommendations(active_playlist)

        if ensemble:
            return recommendations[0:knn]
        return recommendations

    def make_neighborhood_similarity(self, playlist, recommendations=[], knn=5, ensemble=0):
        """
        Make recommendations based on the knn most similar playlists to the active one

        :param playlist: Target playlist
        :param recommendations: already made recommendations in case of padding
        :param knn: number of element to recommend
        :return: recommendations
        :raise ValueError: if the active playlist have no tracks or empty tracks
        """

        knn = knn - len(recommendations)

        possible_recommendations = []
        playlist_tracks = self.db.get_playlist_tracks(playlist)
        playlist_tracks_set = set(playlist_tracks)
        if playlist_tracks == []:
            raise ValueError("playlist is empty")

        playlist_features = [tag for track in playlist_tracks for tag in self.db.get_track_tags(track)]

        playlist_features_set = list(set(playlist_features))
        if len(playlist_features) == 0:
            raise ValueError("playlist have no features!")

        tf_idf_playlist = [(1.0 + math.log(playlist_features.count(tag), 10)) * self.db.get_tag_idf(tag)
                           for tag in playlist_features_set]

        #neighborhood_tracks = set(self.db.get_user_based_collaborative_filtering(playlist))
        neighborhood_content = set([track for playlist in self.db.compute_content_playlists_similarity(playlist) for track in self.db.get_playlist_tracks(playlist)])
        target_tracks = set(self.db.get_target_tracks())
        target_tracks = target_tracks.intersection(neighborhood_content)

        if len(target_tracks) == 0:
            target_tracks = set(self.db.get_target_tracks()).intersection(neighborhood_content)

        for track in target_tracks:
            tags = self.db.get_track_tags(track)
            if track not in playlist_tracks_set and track not in recommendations:

                tf_idf_track = [self.db.get_tag_idf(tag) for tag in tags]

                tag_mask = [float(tag in playlist_features_set) for tag in tags]
                tag_mask_summation = sum(tag_mask)

                # MAP@5 it may be useful if only we knew how to use it
                p_to_k_num = helper.multiply_lists(tag_mask, helper.cumulative_sum(tag_mask))
                p_to_k_den = range(1,len(tag_mask)+1)
                p_to_k = helper.divide_lists(p_to_k_num, p_to_k_den)
                try:
                    map_score = sum(p_to_k) / min(len(playlist_features_set), len(tag_mask))
                except ZeroDivisionError:
                    continue

                precision = tag_mask_summation/len(playlist_features_set)

                try:
                    shrink = math.log(precision, 10)
                except ValueError:
                    continue

                num_cosine_sim = sum([tf_idf_track[tags.index(tag)] * tf_idf_playlist[playlist_features_set.index(tag)] for
                                  tag in tags if tag in playlist_features_set])

                den_cosine_sim = math.sqrt(sum([i ** 2 for i in tf_idf_playlist])) * math.sqrt(
                    sum([i ** 2 for i in tf_idf_track]))

                try:
                    cosine_sim = num_cosine_sim / den_cosine_sim
                except ZeroDivisionError:
                    cosine_sim = 0

                possible_recommendations.append([track, cosine_sim])

        recs = sorted(possible_recommendations, key=itemgetter(1), reverse=True)[0:knn]
        return recommendations + [recommendation for recommendation, value in recs]

    def make_playlist_based_recommendations(self, playlist, target_tracks=[], knn=5, ensemble=0):
        """
        Make Recommendations based on the similarity between playlists

        :param playlist: Target playlist
        :param target_tracks: target tracks to generate prediction for
        :param knn: cardinality of the recommendations
        :return: recommendations
        :raise ValueError: if user had no other playlists or have playlist with some tracks but tagless
        """
        possible_recommendations = []
        playlist_tags = [tag for track in self.db.get_playlist_tracks(playlist) for tag in self.db.get_track_tags(track)]
        playlist_tags_set = set(playlist_tags)
        neighborhood = self.db.compute_collaborative_playlists_similarity(playlist, values="all")
        neighborhood_similarity_sum = math.fsum([value for playlist, value in neighborhood])
        if target_tracks == []:
            target_tracks = self.db.get_target_tracks()

        if playlist_tags == []:
            raise ValueError("user is strunz")

        neighborhood_tracks = [track for playlist, value in neighborhood for track in self.db.get_playlist_tracks(playlist)]
        target_tracks = set(target_tracks).intersection(set(neighborhood_tracks))
        for track in target_tracks:

            prediction = sum([float(track in self.db.get_playlist_tracks(playlist)) * value for playlist, value in neighborhood]) / neighborhood_similarity_sum
            #track_tags = set(self.db.get_track_tags(track))
            #coefficient = neighborhood_tracks.count(track) * len(playlist_tags.intersection(track_tags)) / (float(len(playlist_tags.union(track_tags))))

            possible_recommendations.append([track, prediction])
        recs = sorted(possible_recommendations, key=itemgetter(1), reverse=True)[0:knn]

        if ensemble:
            return recs

        return [recommendation for recommendation, value in recs]


    def make_user_based_recommendations(self, playlist, target_tracks=[], knn=5, ensemble=0):
        """
        Make Recommendations based on the similarity between the tracks of the user who created the target
        playlist and the target track

        :param playlist: Target playlist
        :param target_tracks: target tracks to use
        :param knn: cardinality of the recommendations
        :param ensemble: if used in ensemble
        :return: recommendations
        :raise ValueError: if user had no other playlists or have playlist with some tracks but tagless
        """
        possible_recommendations = []
        playlist_tags = [tag for track in self.db.get_playlist_tracks(playlist) for tag in self.db.get_track_tags(track)]
        playlist_tags_set = set(playlist_tags)
        if playlist_tags == []:
            raise ValueError("user is strunz")
        active_user = self.db.get_playlist_user(playlist)
        neighborhood = self.db.user_user_similarity(active_user)

        neighborhood_similarity_sum = math.fsum([value for user, value in neighborhood])
        if target_tracks == []:
            target_tracks = self.db.get_target_tracks()

        neighborhood_tracks = [track for user, value in neighborhood for track in self.db.get_user_tracks(user)]
        target_tracks = set(target_tracks).intersection(set(neighborhood_tracks))
        for track in target_tracks:
            prediction = sum([self.db.get_user_tracks(user).count(track) * value for user, value in neighborhood])
            #track_tags = set(self.db.get_track_tags(track))
            #coefficient = neighborhood_tracks.count(track) * len(playlist_tags_set.intersection(track_tags)) / (float(len(playlist_tags_set.union(track_tags))))

            possible_recommendations.append([track, prediction])
        recs = sorted(possible_recommendations, key=itemgetter(1), reverse=True)[0:knn]

        if ensemble:
            return recs

        return [recommendation for recommendation, value in recs]

    def make_artists_recommendations(self, playlist, recommendations=[], knn=5, ensemble=0):
        """
        Make recommendations based on the artist of the playlist, if there is enough confidence to affirm
        that a playlist is based on a single dominant artist then target_tracks are choosen from the same artist
        catalogue

        :param playlist: Active playlist
        :param recommendations: Already done recommendations
        :param knn: Number of recommendations to generate
        :return: recommendations
        :raise LookupError: if the playlist is empty
        :raise ValueError: if the playlist have no dominant artist or already includes all song of such artist
        """

        knn -= len(recommendations)

        possible_recommendations = []

        playlist_tracks = self.db.get_playlist_tracks(playlist)
        if playlist_tracks == []:
            raise LookupError("The playlist is empty")

        artists_percentages = []
        scanned = set()
        for track in playlist_tracks:
            artist = self.db.get_artist(track)
            if artist in scanned:
                continue
            scanned.add(artist)
            artist_tracks = self.db.get_artist_tracks(track)
            float_is_in_artist_tracks = [float(track in artist_tracks) for track in playlist_tracks]
            artist_percentage = sum(float_is_in_artist_tracks)/len(playlist_tracks)
            artists_percentages.append([artist, artist_percentage, artist_tracks])

        most_in_artist = max(artists_percentages, key=itemgetter(1))

        if most_in_artist[1] >= 0.66:
            artist_tracks = most_in_artist[2]
        else:
            raise ValueError("No enough confidence")

        target_tracks = self.db.get_target_tracks().difference(playlist_tracks)
        if len(target_tracks) == 0:
            raise ValueError("No artist songs available for selection!")

        #playlist_features = [tag for track in playlist_tracks for tag in self.db.get_track_tags(track)]
        #playlist_features_set = list(set(playlist_features))
        #if len(playlist_features_set) == 0:
        #    raise ValueError("playlist have no features!")

        #tf_idf_playlist = [(1.0 + math.log(playlist_features.count(tag))) * self.db.get_tag_idf(tag) for tag in playlist_features_set]

        for i in target_tracks:

            prediction = sum([self.db.get_item_similarities_alt(i,j) for j in playlist_tracks])
            possible_recommendations.append([i, prediction])

        possible_recommendations = sorted(possible_recommendations, key=itemgetter(1), reverse=True)
        if ensemble:
            return possible_recommendations[0:knn]
        iterator = 0
        while len(recommendations) < 5:
            try:
                item = possible_recommendations[iterator][0]
            except IndexError:
                return recommendations
            if item not in recommendations:
                recommendations.append(item)
            iterator += 1
        return recommendations

    def make_tf_idf_titles_recommendations(self, playlist, target_tracks=[], recommendations=[], knn=5, ensemble=0):
        """
        Make some recommendations based only on the title of the target playlists

        :param playlist: Target playlist
        :param target_tracks: Possible tracks to recommend, default to global target tracks
        :param recommendations: Actual recommendations, needed to not replicate recommendations
        :param knn: Number of recommendations to generate
        :return: List of recommendations added to the list of actual recommendations
        :raises ValueError: if the playlist has no titles
        """

        if target_tracks == []:
            target_tracks = self.db.get_target_tracks()

        knn -= len(recommendations)

        possible_recommendations = []

        playlist_tracks = self.db.get_playlist_tracks(playlist)
        playlist_tracks_set = set(playlist_tracks)

        playlist_titles = self.db.get_playlist_titles(playlist)

        if playlist_titles == []:
            raise ValueError("no titles!")

        tf_idf_titles_playlist = [self.db.get_title_idf(title) for title in playlist_titles]

        for track in target_tracks:
            if track in playlist_tracks_set:
                continue

            titles = self.db.get_track_titles(track)
            titles_set = list(set(titles))

            tf_idf_title = [(1.0 + math.log(titles.count(title), 10)) * self.db.get_title_idf(title) for title in titles_set]

            num_cosine_sim = [tf_idf_title[titles_set.index(title)] * tf_idf_titles_playlist[playlist_titles.index(title)] for
                              title in titles_set if title in playlist_titles]

            den_cosine_sim = math.sqrt(sum([i ** 2 for i in tf_idf_titles_playlist])) * math.sqrt(
                sum([i ** 2 for i in tf_idf_title]))
            try:
                cosine_sim = sum(num_cosine_sim) / den_cosine_sim
            except ZeroDivisionError:
                continue

            possible_recommendations.append([track, cosine_sim])

        if possible_recommendations == []:
            raise ValueError("no recommendations")

        possible_recommendations.sort(key=itemgetter(1), reverse=True)

        if ensemble:
            return possible_recommendations[0:knn]

        return recommendations + [recommendation for recommendation, value in possible_recommendations[0:knn]]


    def combined_top_tag_tfidf_recommendations(self, playlist):
        """
        This function filter knn tracks from the global target tracks with the
         top tag and then apply tf idf recommendations on the filtered list

        :param playlist: Target playlist
        :return: Recommendations
        """
        neighborhood_tracks = self.db.compute_collaborative_playlists_similarity(playlist, tracks_knn=500)
        target_tracks = self.db.get_target_tracks()
        filtered_tracks = self.make_top_tag_recommendations(playlist, target_tracks=target_tracks.intersection(neighborhood_tracks), knn=100)
        other_target_tracks = self.make_collaborative_item_item_recommendations(playlist, target_tracks=filtered_tracks, knn=30)
        try:
            recommendations = set(self.make_tf_idf_recommendations(playlist, target_tracks=other_target_tracks))
            length = len(recommendations)
            iterator = 0
            while length < 5:
                logging.debug("not reached 5 recommendations")
                recommendations.add(filtered_tracks[iterator])
                iterator += 1
                length = len(recommendations)
            return recommendations
        except:
            logging.debug("cannot generate recommendations")
            return filtered_tracks[0:5]

    def combined_tfidf_tags_tfidf_titles_recommendations(self, playlist, knn=50):
        """
        This function filter knn tracks from the global target tracks with the
         tf idf method and then apply tf idf title based on the filtered list

        :param playlist: Target playlist
        :param knn: cardinality of the neighborhood
        :return: Recommendations
        """
        filtered_tracks = self.make_tf_idf_recommendations(playlist, knn=knn)
        return self.make_tf_idf_titles_recommendations(playlist, target_tracks=filtered_tracks)

    def ensemble_recommendations(self, playlist, mask=[1, 0, 1, 0, 0, 0, 0, 0]):
        """
        ensemble method which linearly combine several recommendations method
         predictions into one

        :param playlist: Target playlist
        :return: Recommendations
        """
        possible_recommendations = []
        tracks_playlist = self.db.get_playlist_tracks(playlist)
        titles_playlist = self.db.get_playlist_titles(playlist)
        target_tracks = self.db.get_target_tracks()
        target_tracks = self.db.compute_collaborative_playlists_similarity(playlist, tracks_knn=100)
        '''
        try:
            neighborhood_tracks2 = self.db.compute_content_playlists_similarity(playlist, knn=150)
        except ValueError:
            neighborhood_tracks2 = []
        tracks = neighborhood_tracks.union(neighborhood_tracks2)
        '''
        #target_tracks = target_tracks.intersection(neighborhood_tracks)

        knn = len(target_tracks)
        coefficient = [1, 1, 1, 1, 1, 1, 1, 1]
        coefficient = helper.multiply_lists(coefficient, mask)

        if coefficient[0]:
            try:
                recommendations1 = self.make_tf_idf_recommendations(playlist, target_tracks=target_tracks, knn=knn, ensemble=1)
                normalizing1 = max(recommendations1, key=itemgetter(1))[1]
                recommendations1 = defaultdict(lambda: 0.0, {item[0]: (math.lgamma(item[1]) / float(normalizing1)) for item in recommendations1})
            except:
                recommendations1 = defaultdict(lambda: 0.0, {})
        if coefficient[1]:
            try:
                recommendations2 = self.make_collaborative_item_item_recommendations(playlist, target_tracks=target_tracks, knn=knn, ensemble=1)
                normalizing2 = max(recommendations2, key=itemgetter(1))[1]
                recommendations2 = defaultdict(lambda: 0.0, {item[0]: (math.lgamma(item[1]) / float(normalizing2)) for item in recommendations2})
            except:
                recommendations2 = defaultdict(lambda: 0.0, {})
        if coefficient[2]:
            try:
                recommendations3 =  make_top_tag_recommendations(playlist, target_tracks=target_tracks, knn=knn, ensemble=1, neighborhood_enabled=False)
                normalizing3 = max(recommendations3, key=itemgetter(1))[1]
                recommendations3 = defaultdict(lambda: 0.0, {item[0]: (math.lgamma(item[1]) / float(normalizing3)) for item in recommendations3})
            except:
                recommendations3 = defaultdict(lambda: 0.0, {})

        if coefficient[6]:
            try:
                recommendations7 = self.make_artists_recommendations(playlist, knn=knn, ensemble=1)
                normalizing7 = max(recommendations7, key=itemgetter(1))[1]
                recommendations7 = defaultdict(lambda: 0.0, {item[0]: (item[1] / float(normalizing7)) for item in recommendations7})
            except:
                recommendations7 = defaultdict(lambda: 0.0, {})

        if coefficient[4]:
            try:
                recommendations5 = self.make_tf_idf_titles_recommendations(playlist, knn=knn, ensemble=1)
                normalizing5 = max(recommendations5, key=itemgetter(1))[1]
                recommendations5 = defaultdict(lambda: 0.0, {item[0]: (item[1] / float(normalizing5)) for item in recommendations5})
            except:
                recommendations5 = defaultdict(lambda: 0.0, {})

        if coefficient[3]:
            try:
                recommendations4 = self.make_bad_tf_idf_recommendations_jaccard(playlist, knn=knn, ensemble=1)
                normalizing4 = max(recommendations4, key=itemgetter(1))[1]
                recommendations4 = defaultdict(lambda: 0.0, {item[0]: (item[1] / float(normalizing4)) for item in recommendations4})
            except:
                recommendations4 = defaultdict(lambda: 0.0, {})

        if coefficient[7]:
            try:
                recommendations8 = self.make_neighborhood_similarity(playlist, knn=knn, ensemble=1)
                normalizing8 = max(recommendations8, key=itemgetter(1))[1]
                recommendations8 = defaultdict(lambda: 0.0, {item[0]: (item[1] / float(normalizing8)) for item in recommendations8})
            except:
                recommendations8 = defaultdict(lambda: 0.0, {})

        if coefficient[5]:
            try:
                recommendations6 = self.make_naive_bayes_recommendations(playlist, knn=knn, ensemble=1)
                normalizing6 = max(recommendations6, key=itemgetter(1))[1]
                recommendations6 = defaultdict(lambda: 0.0, {item[0]: (item[1] / float(normalizing6)) for item in recommendations6})
            except:
                recommendations6 = defaultdict(lambda: 0.0, {})
        '''
        possible_recommendations = [[item,  ((recommendations1[item] if coefficient[0] else 0) +
                                            (recommendations2[item] if coefficient[1] else 0) +
                                            (recommendations3[item] if coefficient[2] else 0) +
                                            (recommendations4[item] if coefficient[3] else 0) +
                                            (recommendations5[item] if coefficient[4] else 0) +
                                            (recommendations7[item] if coefficient[6] else 0) +
                                            (recommendations6[item] if coefficient[5] else 0) +
                                            (recommendations8[item] if coefficient[7] else 0)) /
                                            math.sqrt(1+(recommendations1[item]**2 if coefficient[0] else 0) +
                                                        (recommendations2[item]**2 if coefficient[1] else 0) +
                                                        (recommendations3[item]**2 if coefficient[2] else 0) +
                                                        (recommendations4[item]**2 if coefficient[3] else 0) +
                                                        (recommendations5[item]**2 if coefficient[4] else 0) +
                                                        (recommendations7[item]**2 if coefficient[6] else 0) +
                                                        (recommendations6[item]**2 if coefficient[5] else 0) +
                                                        (recommendations8[item]**2 if coefficient[7] else 0))]
                                            for item in target_tracks]


        possible_recommendations = [[item,  (recommendations1[item] if coefficient[0] else 0) ,
                                            (recommendations2[item] if coefficient[1] else 0) ,
                                            (recommendations3[item] if coefficient[2] else 0) ,
                                            (recommendations4[item] if coefficient[3] else 0) ,
                                            (recommendations5[item] if coefficient[4] else 0) ,
                                            (recommendations7[item] if coefficient[6] else 0) ,
                                            (recommendations6[item] if coefficient[5] else 0) ,
                                            (recommendations8[item] if coefficient[7] else 0)]
                                            for item in target_tracks]

        '''
        possible_recommendations = [[item, 0.95*recommendations1[item] * 0.05 * recommendations3[item]] for item in target_tracks]
        possible_recommendations.sort(key=itemgetter(1), reverse=True)

        return [item[0] for item  in possible_recommendations[0:5]]


    def combined_top_tag_tfidf_titles_recommendations(self, playlist, knn=350):
        """
        This function filter knn tracks from the global target tracks with the
         top tag method and then apply tf idf title based on the filtered list

        :param playlist: Target playlist
        :param knn: cardinality of the neighborhood
        :return: Recommendations
        """
        filtered_tracks = self.make_top_tag_recommendations(playlist, knn=knn)
        return self.make_tf_idf_titles_recommendations(playlist, target_tracks=filtered_tracks)

    def combined_tfidf_tfidf_titles_recommendations(self, playlist, knn=100):
        """
        This function filter knn tracks from the global target tracks with the
         tf idf method and then apply tf idf title based on the filtered list

        :param playlist: Target playlist
        :param knn: cardinality of the neighborhood
        :return: Recommendations
        """
        try:
            filtered = self.make_tf_idf_recommendations(playlist, knn=knn)
        except ValueError: #playlists have no features or empty
            filtered = self.make_top_included_recommendations(playlist, knn=knn)
        if len(filtered) < knn: #only for tf_idf
            filtered += self.make_top_tag_recommendations(playlist, recommendations=filtered, knn=knn)
        try:
            return self.make_tf_idf_titles_recommendations(playlist, target_tracks=filtered)
        except ValueError: #playlist have no title
            return filtered[0:5]

    def epoch_iteration(self, learning_rate=0.0005):

        # Get number of available interactions
        numPositiveIteractions = self.db.get_num_interactions()
        learnings=[]
        self.db.init_item_similarities_epoch()

        # Uniform user sampling without replacement
        for _ in xrange(numPositiveIteractions):

            # Sample
            check = True
            while check:
                v_min, v_max = self.db.get_min_max_playlists()
                playlist = random.randint(v_min, v_max)
                playlist_tracks = self.db.get_playlist_tracks(playlist)
                len_playlist_tracks = len(playlist_tracks)
                check = not len_playlist_tracks > 10

            positive_item_id = playlist_tracks[random.randint(0, len_playlist_tracks-1)]
            check = True
            while check:
                v_min, v_max = self.db.get_min_max_tracks()
                negative_item_id = random.randint(v_min, v_max)
                check = negative_item_id in playlist_tracks

            # Prediction
            x_i = sum([self.db.get_item_similarities_epoch(positive_item_id, track) for track in playlist_tracks])
            x_j = sum([self.db.get_item_similarities_epoch(negative_item_id, track) for track in playlist_tracks])

            # Gradient
            x_ij = x_i - x_j

            gradient = 1 / (1 + math.exp(x_ij))

            learnings.append(learning_rate*gradient)
            # Update
            [self.db.set_item_similarities_epoch(positive_item_id, track, learning_rate * gradient) for track in playlist_tracks]

            [self.db.set_item_similarities_epoch(negative_item_id, track, -learning_rate * gradient) for track in playlist_tracks]
        logging.debug("learning rate avg on epoch %f" % helper.mean(learnings))


    def make_collaborative_item_item_recommendations(self, active_playlist, target_tracks=[], recommendations=[], knn=5, ensemble=0):
        """
        Collaborative recommendations which uses pre-computed item to item similarity
        for building the neighborhood and the predictions

        :param active_playlist: Target playlist
        :param target_tracks: target tracks
        :param recommendations: partially filled recommendations list
        :param knn: recommendations to generate
        :param ensemble: if ensemble is ON
        :return: recommendations list
        :raise ValueError: if playlist empty of tracks or all tracks have no features
        """
        tracks = self.db.get_playlist_tracks(active_playlist)

        tags = set([tag for track in tracks for tag in self.db.get_track_tags(track)])
        #artists = [self.db.get_artist(j) for j in tracks]

        if target_tracks == []:
            target_tracks = self.db.get_target_tracks()
        knn -= len(recommendations)
        playlist_tracks = self.db.get_playlist_tracks(active_playlist)
        if playlist_tracks == []:
            raise ValueError("playlist is empty")

        predictions = []

        for i in target_tracks:
            duration = self.db.get_track_duration(i)
            target_track_tags = self.db.get_track_tags(i)
            if i in playlist_tracks or not (duration > 30000 or duration < 0):
                continue
            #prediction = [self.db.get_item_similarities_alt(i,j) for j in playlist_tracks_set]
            prediction = 0
            #jaccard = helper.jaccard(tags, target_track_tags) / len(tags)
            #artist_i = self.db.get_artist(i)
            #artist_sim = sum([self.db.get_artist_similarities(artist_i, artist) for artist in artists])
            for j in playlist_tracks:
                prediction += self.db.get_item_similarities_alt(i,j)

            #tags_sim = sum([self.db.get_tag_similarities(t, target_tag) for t in tags for target_tag in target_track_tags])

            predictions.append([i, prediction])

        recommendations = sorted(predictions, key=itemgetter(1), reverse=True)[0:knn]
        if ensemble:
            return recommendations
        if knn <= 5:
            logging.debug("playlist: %i generated prediction: %s" % (active_playlist, recommendations))

        return [recommendation for recommendation, value in recommendations]

    def make_bad_tf_idf_recommendations_jaccard(self, playlist, target_tracks=[], recommendations=[], knn=5, ensemble=0):
        """
        Bad because of the slowlyness. It's similar to the normal tf idf but takes
        into account every track on the target playlist as single set of tags
        insted of merging all tracks tags into a big list
        :param playlist: Target playlist
        :param target_tracks: target tracks
        :param recommendations: partially filled recommendations list
        :param knn: recommendations to generate
        :return: recommendations list
        :raise ValueError: if playlist empty of tracks or all tracks have no features
        """

        if target_tracks == []:
            target_tracks = self.db.get_target_tracks()

        tot_tags = self.db.get_num_tag()

        knn -= len(recommendations)
        possible_recommendations = []
        playlist_tracks = self.db.get_playlist_tracks(playlist)
        playlist_tracks_set = set(playlist_tracks)
        #playlist_features = [tag for track in playlist_tracks for tag in self.db.get_track_tags(track)]
        #playlist_features_set = set([tag for track in playlist_tracks for tag in self.db.get_track_tags(track)])
        if playlist_tracks == []:
            raise ValueError("playlist is empty")

        tracks_tags = [set(self.db.get_track_tags(track)) for track in playlist_tracks]

        for track in target_tracks:
            '''
            count = 0
            tags = self.db.get_track_tags(track)
            if tags == []:
                continue
            for tag in playlist_features_set:
                if tag in tags:
                    count += 1
            '''
            if track not in playlist_tracks_set and track not in recommendations:
                tags = self.db.get_track_tags(track)
                if len(tags) == 0:
                    continue
                '''
                hemming = helper.hemming_distance(tags, playlist_features_set)
                #jaccard_titles = helper.jaccard(playlist_titles, track_titles)
                jaccard = helper.jaccard(tags, playlist_features_set)
                phi = helper.phi_coefficient(tags, playlist_features, tot_tags)
                tags_len = len(tags)
                euclidean = helper.euclidean_squared_distance(tags, playlist_features_set)
                playlist_features_len = len(playlist_features)
                value_d = helper.square_sum([self.db.get_item_similarities_alt(track,j) for j in playlist_tracks])
                '''
                jaccard_sim = helper.square_sum([helper.jaccard(tags, track_tags) for track_tags in tracks_tags])

                possible_recommendations.append([track, jaccard_sim])
        recs = sorted(possible_recommendations, key=itemgetter(1), reverse=True)[0:knn]
        if ensemble:
            return recs
    	return recommendations + [recommendation for recommendation, value in recs]


    def make_bad_tf_idf_recommendations(self, playlist, target_tracks=[], recommendations=[], knn=5, ensemble=0):
        """
        Bad because of the slowlyness. It's similar to the normal tf idf but takes
        into account every track on the target playlist as single set of tags
        insted of merging all tracks tags into a big list
        :param playlist: Target playlist
        :param target_tracks: target tracks
        :param recommendations: partially filled recommendations list
        :param knn: recommendations to generate
        :return: recommendations list
        :raise ValueError: if playlist empty of tracks or all tracks have no features
        """

        if target_tracks == []:
            target_tracks = (self.db.get_target_tracks())

        knn -= len(recommendations)
        possible_recommendations = []
        playlist_tracks = self.db.get_playlist_tracks(playlist)
        playlist_tracks_set = set(playlist_tracks)
        if playlist_tracks == []:
            raise ValueError("playlist is empty")

        tracks_tags = [self.db.get_track_tags(track) for track in playlist_tracks]
        tf_idf_tracks_tags = [[self.db.get_tag_idf_track(tag) for tag in tags] for tags in tracks_tags]

        if max([len(track_tags) for track_tags in tracks_tags]) == 0:
            raise ValueError("tracks have no feature")

        for track in target_tracks:
            target_track_tags = self.db.get_track_tags(track)

            if track not in playlist_tracks_set and track not in recommendations:
                tags = self.db.get_track_tags(track)
                if tags == []:
                    continue
                tf_idf_track = [self.db.get_tag_idf_track(tag) for tag in tags]

                similarities = []
                for track_tags in tracks_tags:

                    num_cosine_sim = sum([tf_idf_track[tags.index(tag)] * tf_idf_tracks_tags[tracks_tags.index(track_tags)][track_tags.index(tag)] for tag in tags if tag in track_tags])

                    den_cosine_sim = math.sqrt(sum([i**2 for i in tf_idf_tracks_tags[tracks_tags.index(track_tags)]])) * math.sqrt(sum([i**2 for i in tf_idf_track]))

                    try:
                        cosine_sim = num_cosine_sim/den_cosine_sim
                    except ZeroDivisionError:
                        continue
                    similarities.append(cosine_sim)
                value = sum(similarities) / len(playlist_tracks)
                possible_recommendations.append([track, value])
        recs = sorted(possible_recommendations, key=itemgetter(1), reverse=True)[0:knn]
        if ensemble:
            return recs
        return recommendations + [recommendation for recommendation, value in recs]
