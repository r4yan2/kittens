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
import numpy

from libcpp.vector cimport vector
from libcpp cimport bool

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
        logging.debug("testing %s recommendations for playlist %i" % (recommendations, playlist))
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

        # return the triple
        return [map_score, precision, recall]

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
        
        if choice == 20:
            # Get number of available interactions
            numPositiveInteractions = int(self.db.get_num_interactions())
            self.db.init_item_similarities_epoch()
            
            # do some epoch pre-processing on data
            for i in range(1,10):
                logging.debug("epoch %i/9" % i)
                self.epoch_iteration(1000, i)
                self.db.end_epoch_shrink(100)
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
                recommendations = self.make_top_tag_recommendations(target)

            elif choice == 4:
                try:
                    recommendations = self.make_tf_idf_recommendations(target)
                except ValueError: # No tracks or features
                    recommendations = self.make_top_included_recommendations(target)
                if len(recommendations) < 5: # padding needed
                    recommendations = self.make_some_padding(target, recommendations)

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
                    recommendations = self.make_top_included_recommendations(target)
                if len(recommendations) < 5: # padding needed
                    recommendations = self.make_some_padding(target, recommendations)

            elif choice == 7:
                try:
                    recommendations = self.make_tf_idf_titles_recommendations(target)
                except ValueError:
                    try:
                        recommendations = self.make_tf_idf_recommendations(target)
                    except ValueError:
                        recommendations = self.make_top_included_recommendations(target)
                    if len(recommendations) < 5:
                        recommendations = self.make_some_padding(target, recommendations)

            elif choice == 8:
                recommendations = self.combined_tfidf_tags_tfidf_titles_recommendations(target)

            elif choice == 9:
                recommendations = self.combined_top_tag_tfidf_titles_recommendations(target)

            elif choice == 10:
                recommendations = self.combined_tfidf_tfidf_titles_recommendations(target)

            elif choice == 11:
                try:
                    recommendations = self.make_collaborative_item_item_recommendations(target)
                except ValueError:
                    recommendations = self.make_top_included_recommendations(target)
                if len(recommendations) < 5: # if there are not enough artist tracks to recommend
                    recommendations = self.make_some_padding(target, recommendations)

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
                recommendations = self.make_hybrid_recommendations(target)

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
                except ValueError:
                    pass
                if len(recommendations) < 5:
                    recommendations = self.make_some_padding(target, recommendations=recommendations)

            elif choice == 16:
                try:
                    recommendations = self.make_naive_bayes_recommendations(target)
                except ValueError:
                    pass
                if len(recommendations) < 5:
                    recommendations = self.make_some_padding(target, recommendations)

            elif choice == 17:
                try:
                    recommendations = self.make_bad_tf_idf_recommendations(target)
                except ValueError:
                    recommendations = self.make_some_padding(target, recommendations)

            elif choice == 18:
                try:
                    recommendations = self.make_bad_tf_idf_recommendations_jaccard(target)
                except ValueError:
                    recommendations = self.make_some_padding(target, recommendations)

            elif choice == 19:
                try:
                    recommendations = self.make_playlist_based_recommendations(target)
                except ValueError:
                    pass
                if len(recommendations) < 5:
                    recommendations = self.make_some_padding(target, recommendations=recommendations)
                    
            # doing testing things if test mode enabled
            if test:
                test_result = self.check_recommendations(target, recommendations)
                q_out.put([test_result, number, target, len(db.get_playlist_tracks(target))])

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
        methods = [3, 2] # equal to choice parameter
        i = 0
        while len(recommendations) < 5:
            if methods[i] == 3:
                recommendations = self.make_top_tag_recommendations(playlist, recommendations=recommendations)
            elif methods[i] == 2:
                recommendations = self.make_top_included_recommendations(playlist, recommendations=recommendations)
            i = (i + 1) % len(methods)
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

    def make_top_listened_recommendations(self, playlist, target_tracks=[], recommendations=[], knn=5):
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
        knn -= len(recommendations)
        top_listened = self.db.get_top_listened()
        iterator = 0
        count = 0
        already_included = self.db.get_playlist_tracks(playlist)
        while count < knn:
            item = top_listened[iterator][0]
            if item not in already_included and item not in recommendations and item in target_tracks:
                recommendations.append(item)
                count += 1
            iterator += 1

        return recommendations

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

    def make_top_tag_recommendations(self, active_playlist, target_tracks=[], recommendations=[], knn=5, ensemble=0, neighborhood_enabled=True):
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
        
        if neighborhood_enabled:
            if target_tracks != []:
                raise ValueError("target track conflict with neighborhood")
            tracks_knn = 0
            while len(target_tracks) < 5:
                tracks_knn += 50
                target_tracks = self.db.get_target_tracks()
                neighborhood_tracks = self.db.compute_collaborative_playlists_similarity(active_playlist, tracks_knn=tracks_knn)
                target_tracks = target_tracks.intersection(neighborhood_tracks).difference(active_tracks)
        else:
            if target_tracks == []:
                target_tracks = self.get_target_tracks()
        
        knn -= len(recommendations)
        already_included = active_tracks
        active_tags = [tag for track in active_tracks for tag in self.db.get_track_tags(track)]
        active_tags_set = set(active_tags)

        top_tracks = []
        track_playlists_map = self.db.get_track_playlists_map()
        for track in target_tracks: # make the actual recommendation
            tags = self.db.get_track_tags(track)
            matched = active_tags_set.intersection(tags) # calculate the tags which match
            try:
                # calculate first parameter: matched over playlist tags set
                norm_playlist = math.sqrt(len(active_tags_set))
                norm_track = math.sqrt(len(tags))
                value_a = len(matched)/float(norm_playlist*norm_track)
            except ZeroDivisionError:
                value_a = 0
            except ValueError:
                value_a = 0
                
            not_matched = active_tags_set.union(tags).difference(matched)
            # TODO mse?

            try:
                MSD = (1.0 - len(active_tags_set.union(tags).difference(active_tags_set.intersection(tags)))/float(len(active_tags_set.union(tags))))
                value_c = (len(active_tags_set.intersection(tags))/float(len(active_tags_set.union(tags)))) * MSD
            except ValueError:
                value_c = 0
            
            '''
            now it's time to join parameters together, it may be intresting to apply some gradient descent to the curve generated by parameters, this is only a basic sort by multiply all parameters.
            '''

            top_tracks.append([track, value_c * sum([self.db.get_item_similarities_alt(track,j) for j in active_tracks]) * value_a])

        top_tracks.sort(key=itemgetter(1), reverse=True)
        if ensemble:
            return top_tracks[0:knn]
        return recommendations + [recommendation[0] for recommendation in top_tracks[0:knn]]

    def make_tf_idf_recommendations(self, active_playlist, target_tracks=[], recommendations=[], knn=5, ensemble=0, tf_idf="bm25", coefficient="cosine"):
        """
        Make Recommendations based on the tf-idf of the track tags

        :param playlist: Target playlist
        :param target_tracks: Set of target in which choose the random one
        :param recommendations: Set of recommendations already included
        :param knn: Number of items to recommend
        :return: Recommendations
        :raise ValueError: In case playlist have no tracks or tracks with no features
        """
        if target_tracks == []:
            target_tracks = set(self.db.get_target_tracks())

        knn = knn - len(recommendations)

        possible_recommendations = []
        playlist_tracks = self.db.get_playlist_tracks(active_playlist)
        playlist_tracks_set = set(playlist_tracks)
        if playlist_tracks == []:
            raise ValueError("playlist is empty")

        playlist_features = [tag for track in playlist_tracks for tag in self.db.get_track_tags(track)]

        playlist_features_set = list(set(playlist_features))
        if len(playlist_features) == 0:
            raise ValueError("playlist have no features!")

        if tf_idf == "bm25":
            average = self.db.get_average_playlist_tags_count()
            k = 1.2
            b = 0.75
            tf_idf_playlist = [self.db.get_tag_idf(tag) * ((playlist_features.count(tag) * (k + 1)) / (playlist_features.count(tag) + k * (1 - b + b * (len(playlist_features) / average)))) for tag in playlist_features_set]
        elif tf_idf == "normal":
            tf_idf_playlist = [(1.0 + math.log(playlist_features.count(tag), 10)) * self.db.get_tag_idf(tag) for tag in playlist_features_set]

        for track in target_tracks:
            tags = self.db.get_track_tags(track)
            track_duration = self.db.get_track_duration(track)
            if track not in playlist_tracks_set and (track_duration > 30000 or track_duration < 0) and track not in recommendations:

                if tf_idf == "bm25":
                    tf_idf_track = [self.db.get_tag_idf(tag) * ((k + 1) / (1 + k * (1 - b + b * (len(tags) / self.db.get_average_track_tags_count())))) for tag in tags]
                elif tf_idf == "normal":
                     tf_idf_track = [self.db.get_tag_idf(tag) for tag in tags]

                tag_mask = [float(tag in playlist_features_set) for tag in tags]
                tag_mask_summation = sum(tag_mask)

                # MAP@k it may be useful if only we know how to use it
                p_to_k_num = helper.multiply_lists(tag_mask, helper.cumulative_sum(tag_mask))
                p_to_k_den = range(1,len(tag_mask)+1)
                p_to_k = helper.divide_lists(p_to_k_num, p_to_k_den)
                try:
                    map_score = sum(p_to_k) / min(len(playlist_features_set), len(tag_mask))
                    precision = tag_mask_summation/ len(playlist_features_set)
                    recall = tag_mask_summation / len(tags)
                except ZeroDivisionError:
                    continue

                if not (precision and recall and map_score):
                    continue

                if coefficient == "pearson":

                    mean_tfidf_track = sum(tf_idf_track) / len(tf_idf_track)
                    mean_tfidf_playlist = sum(tf_idf_playlist) / len(tf_idf_playlist)

                    numerator = sum([(tf_idf_track[tags.index(tag)] - mean_tfidf_track) *
                                   (tf_idf_playlist[playlist_features_set.index(tag)] - mean_tfidf_playlist)
                                   for tag in tags if tag in playlist_features_set])

                    denominator = math.sqrt(sum([(i - mean_tfidf_playlist) ** 2 for i in tf_idf_playlist]) *
                    sum([(i - mean_tfidf_track) ** 2 for i in tf_idf_track]))

                elif coefficient == "cosine":

                    numerator = sum([tf_idf_track[tags.index(tag)] * tf_idf_playlist[playlist_features_set.index(tag)] for tag in tags if tag in playlist_features_set])

                    denominator = math.sqrt(sum([i ** 2 for i in tf_idf_playlist])) * math.sqrt(sum([i ** 2 for i in tf_idf_track]))

                try:
                    similarity = numerator / denominator
                except ZeroDivisionError:
                    continue

                possible_recommendations.append([track, similarity])

        possible_recommendations.sort(key=itemgetter(1), reverse=True)
        if ensemble:
            return possible_recommendations[0:knn]
        recs = recommendations + [recommendation for recommendation, value in possible_recommendations[0:knn]]
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

        p_in_playlist = len_playlist_tracks/self.db.get_num_tracks()
        playlist_tags_total = float(sum(playlist_tags_counter.values()))

        '''
        for tag in playlist_tags_set:
            p_tag_yes = playlist_tags_counter[tag] * self.db.get_tag_idf(tag) / float(sum(playlist_tags_counter.values()))
            p_yes = sum(playlist_tags_counter.values()) / float(len_playlist_tags_set)
            p_tag = playlist_tags_counter[tag] * self.db.get_tag_idf(tag) / float(len_playlist_tags_set)
            likelihood_map[tag] = (p_tag_yes * p_yes)
        '''
        for track in target_tracks:
            tags = self.db.get_track_tags(track)

            # avg probability
            try:
                probability = helper.product([playlist_tags_counter[tag]/len_playlist_tracks for tag in tags]) * p_in_playlist / helper.product([playlist_tags_counter[tag]/ playlist_tags_total for tag in tags])
            except:
                continue

            possible_recommendations.append([track, probability])


        possible_recommendations.sort(key=itemgetter(1), reverse=True)
        if ensemble:
            return possible_recommendations[0:knn]
        recs = recommendations + [recommendation for recommendation, value in possible_recommendations[0:knn]]
        return recs
    
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
            neighborhood = self.db.get_knn_item_similarities(track, knn)
            
            # TODO fix

            #possible_recommendations.append([track, probability])


        possible_recommendations.sort(key=itemgetter(1), reverse=True)
        if ensemble:
            return possible_recommendations[0:knn]
        recs = recommendations + [recommendation for recommendation, value in possible_recommendations[0:knn]]
        return recs

    def make_hybrid_recommendations(self, active_playlist, recommendations=[], knn=5):
        """
        Hybrid recommendations method which takes into account the number of tracks and titles in a playlist

        :param playlist: Target playlist
        :param recommendations: Actual recommendations made to the playlist
        :param knn: number of recommendations to generate
        :return: the new recommendations
        """
        playlist_tracks = self.db.get_playlist_tracks(active_playlist)
        playlist_titles = self.db.get_titles_playlist(active_playlist)
        active_playlist_creation = self.db.get_created_at_playlist(active_playlist)

        # case: playlist have no tracks and no title
        if playlist_tracks == [] and playlist_titles == []:

            playlists = self.db.get_user_playlists(active_playlist)
            if playlists == [active_playlist]:
                recommendations = self.make_top_included_recommendations(active_playlist)
            owner_playlists = [playlist for playlist in playlists if math.fabs(self.db.get_created_at_playlist(playlist) - active_playlist_creation) < (60 * 60 * 24 * 180)]
            owner_playlists_tracks = [track for playlist in owner_playlists for track in self.db.get_playlist_tracks(playlist)]
            target_tracks = self.db.get_target_tracks()
            predictions = []
            for track in target_tracks:
                duration = self.db.get_track_duration(track)
                if track in playlist_tracks or not (duration > 30000 or duration < 0):
                    continue
                prediction = sum([self.db.get_item_similarities_alt(track,j) for j in owner_playlists_tracks])
                predictions.append([track, prediction])
            predictions.sort(key=itemgetter(1), reverse=True)
            return [recommendation for recommendation, _ in predictions[0:knn]]


        # case: playlist have no tracks
        elif playlist_tracks == []:
            recommendations = self.make_tf_idf_titles_recommendations(active_playlist)

        else: # playlist is complete of title and tracks
            try:
                recommendations = self.make_tf_idf_recommendations(active_playlist)
            except ValueError:
                recommendations = self.make_top_included_recommendations(active_playlist)
            if len(recommendations) < 5:
                recommendations = self.make_some_padding(active_playlist, recommendations=recommendations)
        return recommendations

    def make_neighborhood_similarity(self, playlist, recommendations=[], knn=5):
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

        neighborhood_tracks = set(self.db.get_user_based_collaborative_filtering(playlist))
        neighborhood_content = set([track for playlist in self.db.compute_content_playlists_similarity(playlist) for track in self.db.get_playlist_tracks(playlist)])
        target_tracks = set(self.db.get_target_tracks())
        target_tracks = target_tracks.intersection(neighborhood_content.intersection(neighborhood_tracks))

        if len(target_tracks) == 0:
            target_tracks = set(self.db.get_target_tracks()).intersection(neighborhood_content)

        for track in target_tracks:
            tags = self.db.get_track_tags(track)
            track_duration = self.db.get_track_duration(track)
            if track not in playlist_tracks_set and (track_duration > 30000 or track_duration < 0) and track not in recommendations:

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
                    shrink = math.log(precision, 10) * 35
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

        possible_recommendations.sort(key=itemgetter(1), reverse=True)
        recs = recommendations + [recommendation for recommendation, value in possible_recommendations[0:knn]]
        return recs

    def make_playlist_based_recommendations(self, playlist, target_tracks=[], knn=5):
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
        possible_recommendations.sort(key=itemgetter(1), reverse=True)
        recs = [recommendation for recommendation, value in possible_recommendations[0:knn]]
        return recs


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
            prediction = sum([self.db.get_user_tracks(user).count(track) * value for user, value in neighborhood]) / neighborhood_similarity_sum
            #track_tags = set(self.db.get_track_tags(track))
            #coefficient = neighborhood_tracks.count(track) * len(playlist_tags.intersection(track_tags)) / (float(len(playlist_tags.union(track_tags))))

            possible_recommendations.append([track, prediction])
        if ensemble:
            return possible_recommendations[0:knn]
        possible_recommendations.sort(key=itemgetter(1), reverse=True)

        recs = [recommendation for recommendation, value in possible_recommendations[0:knn]]
        return recs

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

        target_tracks = self.db.get_target_tracks().intersection(artist_tracks).difference(playlist_tracks)
        if len(target_tracks) == 0:
            raise ValueError("No artist songs available for selection!")

        #playlist_features = [tag for track in playlist_tracks for tag in self.db.get_track_tags(track)]
        #playlist_features_set = list(set(playlist_features))
        #if len(playlist_features_set) == 0:
        #    raise ValueError("playlist have no features!")

        #tf_idf_playlist = [(1.0 + math.log(playlist_features.count(tag))) * self.db.get_tag_idf(tag) for tag in playlist_features_set]

        for i in target_tracks:
            duration = self.db.get_track_duration(i)
            if (duration <= 30000 and duration >= 0):
                continue
            prediction = sum([self.db.get_item_similarities_alt(i,j) for j in playlist_tracks])
            possible_recommendations.append([i, prediction])

        possible_recommendations.sort(key=itemgetter(1), reverse=True)
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

        playlist_titles = self.db.get_titles_playlist(playlist)

        playlist_created_at = self.db.get_created_at_playlist(playlist)

        if playlist_titles == []:
            raise ValueError("no titles!")

        tf_idf_titles_playlist = [self.db.get_title_idf(title) for title in playlist_titles]

        similarities = []
        playlists = self.db.get_playlists()
        for playlist in playlists:

            if math.fabs(self.db.get_created_at_playlist(playlist) - playlist_created_at) > (60*60*24*365*2):
                continue

            titles = set(self.db.get_titles_playlist(playlist))

            coefficient = helper.jaccard(titles, playlist_titles)

            similarities.append([playlist, coefficient])
        similarities.sort(key=itemgetter(1), reverse=True)
        neighborhood = [playlist for playlist, _ in similarities[0:50]]
        neighborhood_tracks = [track for playlist in neighborhood for track in self.db.get_playlist_tracks(playlist) if track in target_tracks]

        possible_recommendations = Counter(neighborhood_tracks).items()
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

    def ensemble_recommendations(self, playlist):
        """
        ensemble method which linearly combine several recommendations method
         predictions into one

        :param playlist: Target playlist
        :return: Recommendations
        """
        possible_recommendations = []
        target_tracks = self.db.get_target_tracks()
        neighborhood_tracks = self.db.compute_collaborative_playlists_similarity(playlist, tracks_knn=150)
        target_tracks = target_tracks.intersection(neighborhood_tracks)

        knn = len(target_tracks)

        try:
            recommendations1 = self.make_tf_idf_recommendations(playlist, target_tracks=target_tracks, knn=knn, ensemble=1)
            normalizing1 = max(recommendations1, key=itemgetter(1))[1]
            recommendations1 = defaultdict(lambda: 0.0, {item[0]: (item[1] / float(normalizing1)) for item in recommendations1})
        except:
            recommendations1 = defaultdict(lambda: 0.0, {})

        try:
            recommendations2 = self.make_collaborative_item_item_recommendations(playlist, target_tracks=target_tracks, knn=knn, ensemble=1)
            normalizing2 = max(recommendations2, key=itemgetter(1))[1]
            recommendations2 = defaultdict(lambda: 0.0, {item[0]: (item[1] / float(normalizing2)) for item in recommendations2})
        except:
            recommendations2 = defaultdict(lambda: 0.0, {})

        try:
            recommendations3 = self.make_top_tag_recommendations(playlist, target_tracks=target_tracks, knn=knn, ensemble=1, neighborhood_enabled=False)
            normalizing3 = max(recommendations3, key=itemgetter(1))[1]
            recommendations3 = defaultdict(lambda: 0.0, {item[0]: (item[1] / float(normalizing3)) for item in recommendations3})
        except:
            recommendations3 = defaultdict(lambda: 0.0, {})

        try:
            recommendations7 = self.make_artists_recommendations(playlist, knn=knn, ensemble=1)
            normalizing7 = max(recommendations7, key=itemgetter(1))[1]
            recommendations7 = defaultdict(lambda: 0.0, {item[0]: (item[1] / float(normalizing7)) for item in recommendations7})
        except:
            recommendations7 = defaultdict(lambda: 0.0, {})
            
        try:
            recommendations5 = self.make_tf_idf_titles_recommendations(playlist, knn=knn, ensemble=1)
            normalizing5 = max(recommendations5, key=itemgetter(1))[1]
            recommendations5 = defaultdict(lambda: 0.0, {item[0]: (item[1] / float(normalizing5)) for item in recommendations5})
        except:
            recommendations5 = defaultdict(lambda: 0.0, {})

        a = 0.40
        b = 0.25
        c = 0.20
        e = 0.05
        g = 0.10

        possible_recommendations = [[item, a * recommendations1[item] + c * recommendations3[item] + b * recommendations2[item] +  e * recommendations5[item] + g * recommendations7[item]] for item in target_tracks]
        possible_recommendations.sort(key=itemgetter(1), reverse=True)

        return [item for item, value in possible_recommendations[0:5]]


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

    def epoch_iteration(self, int numPositiveInteractions, int epoch, double learning_rate=0.005):

        cdef vector[double] learnings = []
        # Uniform user sampling without replacement
        
        cdef vector[int] playlists = list(self.db.get_target_playlists())
        cdef vector[int] tracks = self.db.get_tracks()
        cdef bool check = True
        cdef int playlist
        cdef int len_playlist_tracks
        cdef int positive_item_id
        cdef int negative_item_id
        cdef int x_i
        cdef int x_j
        cdef int x_ij
        cdef int gradient
        
        for _ in range(numPositiveInteractions):

            # Sample
            while check:
                playlist = random.choice(playlists)
                playlist_tracks = self.db.get_playlist_tracks(playlist)
                len_playlist_tracks = len(playlist_tracks)
                check = not len_playlist_tracks > 10

            positive_item_id = random.choice(playlist_tracks)
            check = True
            while check:

                negative_item_id = random.choice(tracks)
                check = negative_item_id in playlist_tracks

            np_playlist_tracks = numpy.array(playlist_tracks)

            # Prediction
            x_i = self.db.get_item_similarities_epoch(epoch, positive_item_id, np_playlist_tracks).sum()
            x_j = self.db.get_item_similarities_epoch(epoch, negative_item_id, np_playlist_tracks).sum()

            # Gradient
            x_ij = x_i - x_j

            gradient = 1 / (1 + math.exp(x_ij))

            learnings.push_back(learning_rate*gradient)

            # Update
            self.db.set_item_similarities_epoch(positive_item_id, np_playlist_tracks, learning_rate * gradient)
            self.db.null_item_similarities_epoch(positive_item_id, positive_item_id)

            self.db.set_item_similarities_epoch(negative_item_id, np_playlist_tracks, -learning_rate * gradient)
            self.db.null_item_similarities_epoch(negative_item_id, negative_item_id)
        logging.debug("learning rate avg on epoch %f" % helper.mean(learnings))


    def make_collaborative_item_item_recommendations(self, active_playlist, target_tracks=[], recommendations=[], knn=5, ensemble=0):
        """
        Collaborative recommendations which uses pre-computed item to item similarity
        for building the neighborhood and the predictions

        :param playlist: Target playlist
        :param target_tracks: target tracks
        :param recommendations: partially filled recommendations list
        :param knn: recommendations to generate
        :return: recommendations list
        :raise ValueError: if playlist empty of tracks or all tracks have no features
        """

        if target_tracks == []:
            target_tracks = list(self.db.get_target_tracks())

        knn -= len(recommendations)

        predictions = []
        
        URM = self.db.get_URM()
        playlist_tracks = URM[active_playlist]
        scores = playlist_tracks.dot(self.db.get_item_similarities_map()).toarray().ravel()
    
        # remove seen
        seen = URM.indices[URM.indptr[active_playlist]:URM.indptr[active_playlist + 1]]
        scores[seen] = -numpy.inf
        
        # considering only target tracks
        scores[target_tracks] += 10000
        
        ranking = scores.argsort()[::-1][:knn]
        
        return ranking

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

        knn -= len(recommendations)
        possible_recommendations = []
        playlist_tracks = self.db.get_playlist_tracks(playlist)
        playlist_tracks_set = set(playlist_tracks)
        if playlist_tracks == []:
            raise ValueError("playlist is empty")

        tracks_tags = [set(self.db.get_track_tags(track)) for track in playlist_tracks]

        for track in target_tracks:
            track_duration = self.db.get_track_duration(track)
            if track not in playlist_tracks_set and (track_duration > 30000 or track_duration < 0) and track not in recommendations:
                tags = set(self.db.get_track_tags(track))
                if len(tags) == 0:
                    continue

                jaccard_sim = sum([helper.jaccard(tags, track_tags) for track_tags in tracks_tags])

                possible_recommendations.append([track, jaccard_sim])
        recommendations = sorted(possible_recommendations, key=itemgetter(1), reverse=True)[0:knn]
        if ensemble:
            return recommendations
        return [recommendation for recommendation, value in recommendations]


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
            track_duration = self.db.get_track_duration(track)
            if track not in playlist_tracks_set and (track_duration > 30000 or track_duration < 0) and track not in recommendations:
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
                value = sum(similarities)
                possible_recommendations.append([track, value])
        possible_recommendations.sort(key=itemgetter(1), reverse=True)
        if ensemble:
            return possible_recommendations[0:knn]
        recommendations = sorted(possible_recommendations, key=itemgetter(1), reverse=True)[0:knn]
        return [recommendation for recommendation, value in recommendations]
