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

class Recommender:
    def __init__(self, db=None):
        if db:
            self.db = db # only useful for debug

    def check_recommendations(self, playlist, recommendations):
        test_set = self.db.get_playlist_relevant_tracks(playlist)
        test_set_length = len(test_set)
        if len(recommendations) < 5:
            raise ValueError('Recommendations list have less than 5 recommendations')
        is_relevant = [item in test_set for item in recommendations]
        is_relevant_length = len(is_relevant)

        # MAP@5
        p_to_k_num = helper.multiply_lists(is_relevant, helper.cumulative_sum([float(i) for i in is_relevant]))
        p_to_k_den = range(1,is_relevant_length+1)
        p_to_k = helper.divide_lists(p_to_k_num, p_to_k_den)
        try:
            map_score = sum(p_to_k) / min(test_set_length, is_relevant_length)
        except ZeroDivisionError:
            map_score = 0

        # Precision
        try:
            precision = sum(is_relevant)/float(is_relevant_length)
        except ZeroDivisionError:
            precision = 0

        # Recall
        try:
            recall = sum(is_relevant)/float(test_set_length)
        except ZeroDivisionError:
            recall = 0

        # return the triple
        return [map_score, precision, recall]


    def run(self, choice, db, q_in, q_out, test, number):

        # Retrieve the db from the list of arguments
        self.db = db

        # main loop for the worker
        while True:

            # getting the data from the queue in
            (identifier, target) = q_in.get()

            # if is the end stop working
            if target == -1:
                break
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
                recommendations = self.combined_top_tag_tfidf_recommendations(target)

            elif choice == 6:
                recommendations = self.combined_tfidf_top_tag_recommendations(target)

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
                    recommendations = self.make_bad_tf_idf_recommendations(target)
                except ValueError:
                    recommendations = self.make_top_included_recommendations(target)

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

            # doing testing things if test mode enabled
            if test:
                test_result = self.check_recommendations(target, recommendations)
                q_out.put([test_result, number, target])

            else:
                # else put the result into the out queue
                q_out.put([identifier, target, recommendations])

    def make_some_padding(self, playlist, recommendations, knn=5):
        """
        Make some padding when needed

        :param playlist: target playlist
        :param recommendations: actual recommended tracks
        :return: the full list of tracks recommended to the target playlist
        """
        methods = [7, 2] # equal to choice parameter
        i = 0
        remaining = knn - len(recommendations)
        while remaining > 0:
            if methods[i] == 7:
                recommendations = self.make_tf_idf_titles_recommendations(playlist, recommendations=recommendations, knn=remaining)
            elif methods[i] == 2:
                recommendations = self.make_top_included_recommendations(playlist, recommendations=recommendations, knn=remaining)
            remaining = knn - len(recommendations)
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

    def make_top_tag_recommendations(self, active_playlist, target_tracks=[], recommendations=[], knn=5):
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
        if target_tracks == []:
            target_tracks = self.db.get_target_tracks()

        knn -= len(recommendations)
        active_tracks = self.db.get_playlist_tracks(active_playlist) # get already included tracks
        already_included = active_tracks
        active_tags = [tag for track in active_tracks for tag in self.db.get_track_tags(track)]
        active_tags_set = set(active_tags)

        top_tracks = []
        track_playlists_map = self.db.get_track_playlists_map()

        for track in target_tracks: # make the actual recommendation
            track_duration = self.db.get_track_duration(track) # get the track length
            tags = self.db.get_track_tags(track)
            if track not in already_included and track not in recommendations and (track_duration > 30000 or track_duration < 0):
                matched = filter(lambda x: x in active_tags_set, tags) # calculate the tags which match
                try:
                    # calculate first parameter: matched over playlist tags set
                    norm_playlist = math.sqrt(len(active_tags_set))
                    norm_track = math.sqrt(len(tags))
                    value_a = len(matched)/float(norm_playlist*norm_track)
                except ZeroDivisionError:
                    value_a = 0
                except ValueError:
                    value_a = 0
                try:
                    # calculate second parameter: matched over track tags
                    value_b = len(matched)/float(len(tags))
                except ZeroDivisionError:
                    value_b = 0

                top_value = len(track_playlists_map[track]) # get the value from the top-included

                top_tracks.append([track, value_a, value_b, top_value]) # joining all parameters together

        top_tracks.sort(key=itemgetter(1, 2, 3), reverse=True)
        return recommendations + [recommendation[0] for recommendation in top_tracks[0:knn]]

    def make_tf_idf_recommendations(self, playlist, target_tracks=[], recommendations=[], knn=5):
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
            target_tracks = self.db.get_target_tracks()

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
        k = 2.0
        b = 0.75
        average = self.db.get_average_playlist_length()
        tf_idf_playlist = [self.db.get_tag_idf(tag) * ((playlist_features.count(tag) * (k + 1)) / (playlist_features.count(tag) + k * (1 - b + b * (len(playlist_features) / average)))) 
                           for tag in playlist_features_set]

        for track in target_tracks:
            tags = self.db.get_track_tags(track)
            track_duration = self.db.get_track_duration(track)
            if track not in playlist_tracks_set and (track_duration > 30000 or track_duration < 0) and track not in recommendations:

                tf_idf_track = [self.db.get_tag_idf_track(tag) * ((k + 1) / (1 + k * (1 - b + b * (len(tags) / self.db.get_average_tags_length())))) for tag in tags]

                tag_mask = [float(tag in playlist_features_set) for tag in tags]
                tag_mask_summation = sum(tag_mask)

                # MAP@k it may be useful if only we knew how to use it
                p_to_k_num = helper.multiply_lists(tag_mask, helper.cumulative_sum(tag_mask))
                p_to_k_den = range(1,len(tag_mask)+1)
                p_to_k = helper.divide_lists(p_to_k_num, p_to_k_den)
                try:
                    map_score = sum(p_to_k) / min(len(playlist_features_set), len(tag_mask))
                except ZeroDivisionError:
                    continue

                precision = tag_mask_summation/len(playlist_features_set)

                try:
                    shrink = math.log(map_score, 10) * 100
                except ValueError:
                    continue

                '''
                #Pearson correlation coefficient

                mean_tfidf_track = sum(tf_idf_track) / len(tf_idf_track)

                mean_tfidf_playlist = sum(tf_idf_playlist) / len(tf_idf_playlist)

                num_pearson_sim = [(tf_idf_track[tags.index(tag)] - mean_tfidf_track) *
                                   (tf_idf_playlist[playlist_features_set.index(tag)] - mean_tfidf_playlist)
                                   for tag in tags if tag in playlist_features_set]


                den_pearson_sim = math.sqrt(sum([(i - mean_tfidf_playlist) ** 2 for i in tf_idf_playlist]) *
                    sum([(i - mean_tfidf_track) ** 2 for i in tf_idf_track]))

                try:
                    pearson_sim = sum(num_pearson_sim)/ (den_pearson_sim)
                except ZeroDivisionError:
                    pearson_sim = 0
                '''
                #Cosine similarity

                num_cosine_sim = sum([tf_idf_track[tags.index(tag)] * tf_idf_playlist[playlist_features_set.index(tag)]
                                  for tag in tags if tag in playlist_features_set])

                den_cosine_sim = math.sqrt(sum([i ** 2 for i in tf_idf_playlist])) * math.sqrt(
                    sum([i ** 2 for i in tf_idf_track]))

                try:
                    cosine_sim = num_cosine_sim / (den_cosine_sim - shrink)
                except ZeroDivisionError:
                    cosine_sim = -1000000000

                possible_recommendations.append([track, cosine_sim])

        possible_recommendations.sort(key=itemgetter(1), reverse=True)
        recs = recommendations + [recommendation for recommendation, value in possible_recommendations[0:knn]]
        return recs

    def make_hybrid_recommendations(self, playlist, recommendations=[], knn=5):
        """
        Hybrid recommendations method which takes into account the number of tracks and titles in a playlist

        :param playlist: Target playlist
        :param recommendations: Actual recommendations made to the playlist
        :param knn: number of recommendations to generate
        :return: the new recommendations
        """
        playlist_tracks = self.db.get_playlist_tracks(playlist)
        playlist_titles = self.db.get_titles_playlist(playlist)

        if playlist_tracks == [] and playlist_titles == []: # playlist have no tracks or title
            try:
                recommendations = self.make_user_based_recommendations(playlist)
            except ValueError:
                recommendations = self.make_top_included_recommendations(playlist)
        elif playlist_tracks == []: # playlist have no tracks
            recommendations = self.make_tf_idf_titles_recommendations(playlist)
        else: # playlist is complete of title and tracks
            try:
                recommendations = self.make_tf_idf_recommendations(playlist)
            except ValueError:
                recommendations = self.make_top_included_recommendations(playlist)
            if len(recommendations) < 5:
                recommendations = self.make_some_padding(playlist, recommendations=recommendations)
        return recommendations

    def make_neighborhood_similarity(self, playlist, recommendations=[], knn=5):
        """
        Make recommendations based on the knn most similar playlists to the active one

        :param playlist: Target playlist
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

        neighborhood = self.db.compute_playlists_similarity(playlist)
        neighborhood_tracks = list(set([track for item in neighborhood for track in self.db.get_playlist_tracks(item)]))
        print "playlist:",playlist,"neighborhood length", len(neighborhood)
        target_tracks = set(self.db.get_target_tracks())
        target_tracks = [track for track in neighborhood_tracks if track in target_tracks]
        print "target_tracks length", len(target_tracks)

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


    def make_user_based_recommendations(self, playlist):
        """
        Make Recommendations based on the similarity between the user who created the target
        platlist and the users which included the target track

        :param playlist: Target playlist
        :return: recommendations
        :raise ValueError: if user had no other playlists or have playlist with some tracks but tagless
        """
        knn = 5
        possible_recommendations = []
        user_tracks = self.db.get_playlist_user_tracks(playlist)
        target_tracks = []
        if target_tracks == []:
            target_tracks = self.db.get_target_tracks()

        if user_tracks == []:
            raise ValueError("user is strunz")
        user_tracks_set = set(user_tracks)

        user_features = [tag for track in user_tracks for tag in self.db.get_track_tags(track)]

        user_features_set = list(set(user_features))
        if len(user_features_set) == 0:
            raise ValueError("user have no features!")

        tf_idf_user = [(1.0 + math.log(user_features.count(tag), 10)) * self.db.get_tag_idf(tag)
                           for tag in user_features_set]

        for track in target_tracks:
            tags = self.db.get_track_tags(track)
            track_duration = self.db.get_track_duration(track)
            if track not in user_tracks_set and (track_duration > 30000 or track_duration < 0):

                tf_idf_track = [1.0 * self.db.get_tag_idf(tag) for tag in tags]

                tag_mask = [float(tag in user_features_set) for tag in tags]

                num_cosine_sim = [tf_idf_track[tags.index(tag)] * tf_idf_user[user_features_set.index(tag)] for
                                  tag in tags if tag in user_features_set]


                den_cosine_sim = math.sqrt(sum([i ** 2 for i in tf_idf_user])) * math.sqrt(
                    sum([i ** 2 for i in tf_idf_track]))
                try:
                    shrink = math.log(sum(tag_mask)/len(user_features_set), 10) * 35
                except ValueError:
                    shrink = -100
                try:
                    cosine_sim = sum(num_cosine_sim) / (den_cosine_sim - shrink)
                except ZeroDivisionError:
                    cosine_sim = 0

                possible_recommendations.append([track, cosine_sim])

        possible_recommendations.sort(key=itemgetter(1), reverse=True)
        recs = [recommendation for recommendation, value in possible_recommendations[0:knn]]
        return recs

    def make_artists_recommendations(self, playlist, recommendations=[], knn=5):
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
        for track in playlist_tracks:
            artist_tracks = self.db.get_artist_tracks(track)
            float_is_in_artist_tracks = [float(track in artist_tracks) for track in playlist_tracks]
            artist_percentage = sum(float_is_in_artist_tracks)/len(playlist_tracks)
            artist_id = self.db.get_artist(track)
            artists_percentages.append([artist_id, artist_percentage, artist_tracks])

        artists_percentages.sort(key = itemgetter(1),reverse = True)
        most_in_artist = artists_percentages[0][1]

        if most_in_artist > 0.66:
            artist_tracks = artists_percentages[0][2]
        else:
            raise ValueError("The playlist have no dominant artist")

        target_tracks = helper.diff_list(artist_tracks, playlist_tracks)
        if target_tracks == []:
            raise ValueError("The playlist contains all the songs of the artist!")

        playlist_features = [tag for track in playlist_tracks for tag in self.db.get_track_tags(track)]
        playlist_features_set = list(set(playlist_features))
        if len(playlist_features_set) == 0:
            raise ValueError("playlist have no features!")

        tf_idf_playlist = [(1.0 + math.log(playlist_features.count(tag))) * self.db.get_tag_idf(tag) for tag in playlist_features_set]

        for track in target_tracks:
            tags = self.db.get_track_tags(track)
            track_duration = self.db.get_track_duration(track)
            if (track_duration > 30000 or track_duration < 0):
                tf_idf_track = [self.db.get_tag_idf(tag) for tag in tags]

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

    def make_tf_idf_titles_recommendations(self, playlist, target_tracks=[], recommendations=[], knn=5):
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

        if playlist_titles == []:
            raise ValueError("no titles!")

        tf_idf_titles_playlist = [self.db.get_title_idf(title) for title in playlist_titles]

        for track in target_tracks:
            track_duration = self.db.get_track_duration(track)
            if track not in playlist_tracks_set and (track_duration > 30000 or track_duration < 0):

                titles = self.db.get_titles_track(track)
                titles_set = list(set(titles))

                tf_idf_title = [(1.0 + math.log(titles.count(title), 10)) * self.db.get_title_idf(title) for title in titles_set]

                num_cosine_sim = [tf_idf_title[titles_set.index(title)] * tf_idf_titles_playlist[playlist_titles.index(title)] for
                                  title in titles_set if title in playlist_titles]

                den_cosine_sim = math.sqrt(sum([i ** 2 for i in tf_idf_titles_playlist])) * math.sqrt(
                    sum([i ** 2 for i in tf_idf_title]))
                try:
                    cosine_sim = sum(num_cosine_sim) / den_cosine_sim
                except ZeroDivisionError:
                    cosine_sim = 0

                possible_recommendations.append([track, cosine_sim])

        possible_recommendations.sort(key=itemgetter(1), reverse=True)
        logging.debug("top5 recomm for %i : (item,value) %s" % (playlist, possible_recommendations[0:5]))
        return recommendations + [recommendation for recommendation, value in possible_recommendations[0:knn]]


    def combined_top_tag_tfidf_recommendations(self, playlist, knn=125):
        """
        This function filter knn tracks from the global target tracks with the
         top tag and then apply tf idf recommendations on the filtered list

        :param playlist: Target playlist
        :param knn: cardinality of the neighborhood
        :return: Recommendations
        """
        filtered_tracks = self.make_top_tag_recommendations(playlist, knn=knn)
        return self.make_tf_idf_recommendations(playlist, target_tracks=filtered_tracks)

    def combined_tfidf_tags_tfidf_titles_recommendations(self, playlist, knn = 50):
        """
        This function filter knn tracks from the global target tracks with the
         tf idf method and then apply tf idf title based on the filtered list

        :param playlist: Target playlist
        :param knn: cardinality of the neighborhood
        :return: Recommendations
        """
        filtered_tracks = self.make_tf_idf_recommendations(playlist, knn=knn)
        return self.make_tf_idf_titles_recommendations(playlist, target_tracks=filtered_tracks)

    def combined_tfidf_top_tag_recommendations(self, playlist, knn=125):
        """
        This function filter knn tracks from the global target tracks with the
         tf idf method and then apply top tag method on the filtered list

        :param playlist: Target playlist
        :param knn: cardinality of the neighborhood
        :return: Recommendations
        """
        filtered_tracks = self.make_tf_idf_recommendations(playlist, knn=knn)
        return self.make_top_tag_recommendations(playlist, target_tracks=filtered_tracks)

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

    def make_bad_tf_idf_recommendations(self, playlist, target_tracks=[], recommendations=[], knn=5):
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

        tracks_tags = [self.db.get_track_tags(track) for track in playlist_tracks]
        tf_idf_tracks_tags = [[self.db.get_tag_idf(tag) for tag in tags] for tags in tracks_tags]

        if max([len(track_tags) for track_tags in tracks_tags]) == 0:
            raise ValueError("tracks have no feature")
        
        normalized_rating = [self.db.get_normalized_rating(playlist, track) for track in playlist_tracks]

        for track in target_tracks:
            track_duration = self.db.get_track_duration(track)
            if track not in playlist_tracks_set and (track_duration > 30000 or track_duration < 0) and track not in recommendations:
                tags = self.db.get_track_tags(track)
                if tags == []:
                    continue
                tf_idf_track = [self.db.get_tag_idf(tag) for tag in tags]

                similarities = []
                for track_tags in tracks_tags:
                    
                    num_cosine_sim = sum([tf_idf_track[tags.index(tag)] *
                           tf_idf_tracks_tags[tracks_tags.index(track_tags)][track_tags.index(tag)] *
                           math.log(1.0 - (math.fabs(tags.index(tag) - track_tags.index(tag))/len(tags)),10)
                           for tag in tags if tag in track_tags])

                    den_cosine_sim = math.sqrt(sum([i**2 for i in tf_idf_tracks_tags[tracks_tags.index(track_tags)]])) * math.sqrt(sum([i**2 for i in tf_idf_track]))

                    try:
                        cosine_sim = num_cosine_sim/den_cosine_sim
                    except ZeroDivisionError:
                        cosine_sim = -1000000000
                    similarities.append(cosine_sim)
                numerator = sum([similarity * normalized_rating[similarities.index(similarity)] for similarity in similarities])
                denominator = sum(similarities)
                try:
                    value = numerator/denominator
                except ZeroDivisionError:
                    value = 0
                possible_recommendations.append([track, value])
        possible_recommendations.sort(key=itemgetter(1), reverse=True)
        iterator = 0
        while len(recommendations) < knn:
            item = possible_recommendations[iterator][0]
            if item not in recommendations:
                recommendations.append(item)
            iterator += 1
        return recommendations
