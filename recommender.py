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
            if choice == 0:
                recommendations = self.make_random_recommendations(target)
            elif choice == 1:
                recommendations = self.make_top_listened_recommendations(target)
            elif choice == 2:
                recommendations = self.make_top_included_recommendations(target)
            elif choice == 3:
                recommendations = self.make_top_tag_recommendations(target, self.db.get_target_tracks(), [])
            elif choice == 4:
                try:
                    recommendations = self.make_tf_idf_recommendations(target, self.db.get_target_tracks(), [])
                except ValueError:
                    recommendations = self.make_top_included_recommendations(target)
                if len(recommendations) < 5:
                    recommendations.extend(self.make_top_tag_recommendations(target, self.db.get_target_tracks(), recommendations))
            elif choice == 5:
                recommendations = self.combined_top_tag_tfidf_recommendations(target)
            elif choice == 6:
                recommendations = self.combined_tfidf_top_tag_recommendations(target)
            elif choice == 7:
                recommendations = self.make_tf_idf_titles_recommendations(target, self.db.get_target_tracks(), 5)
            elif choice == 8:
                recommendations = self.combined_tfidf_tags_tfidf_titles_recommendations(target)
            elif choice == 9:
                recommendations = self.combined_top_tag_tfidf_titles_recommendations(target)
            elif choice == 10:
                recommendations = self.combined_tfidf_tfidf_titles_recommendations(target)
            elif choice == 11:
                recommendations = self.make_bad_tf_idf_recommendations(target)
            elif choice == 12:
                try:
                    recommendations = self.make_artists_recommendations(target, 5)
                except LookupError:
                    # TODO implement user-targeted recommendations
                    recommendations = self.make_top_included_recommendations(target)
                except ValueError: # no dominant artist
                    try:
                        recommendations = self.make_tf_idf_recommendations(target, self.db.get_target_tracks(), [])
                    except ValueError: # this may happend when the playlist have 1-2 tracks with no features (fuck it)
                        recommendations = self.make_top_included_recommendations(target)
                if len(recommendations) < 5: # if there are not enough artist tracks to recommend or if the tracks have a strage avg duration
                    recommendations.extend(self.make_tf_idf_recommendations(target, self.db.get_target_tracks(), recommendations))
                if len(recommendations) < 5:
                    recommendations.extend(self.make_top_tag_recommendations(target, self.db.get_target_tracks(), recommendations))

            # doing testing things if test mode enabled
            if test:
                test_result = self.check_recommendations(target, recommendations)
                q_out.put([test_result, number])

            else:
                # else put the result into the out queue
                q_out.put([identifier, target, recommendations])

    def make_random_recommendations(self, playlist):
        """
        take 5 random tracks and recommend them

        :param playlist:
        :return:
        """
        recommendations = []
        count = 0
        already_included = self.db.get_playlist_tracks(playlist)
        target_tracks = self.db.get_target_tracks()
        num_target_tracks = len(target_tracks)
        while count < 5:
            track = random.randint(0, num_target_tracks)
            if (track not in already_included) and (track not in recommendations):
                recommendations.append(target_tracks[track])
                count += 1
        return recommendations

    def make_top_listened_recommendations(self, playlist):
        """
        recommend the top listened tracks

        :param playlist:
        :return:
        """
        recommendations = []
        top_listened = self.db.get_top_listened()
        iterator = 0
        count = 0
        already_included = self.db.get_playlist_tracks(playlist)
        target_tracks = self.db.get_target_tracks()
        while count < 5:
            item = top_listened[iterator][0]
            if (item not in already_included) and (item not in recommendations) and (item in target_tracks):
                recommendations.append(item)
                count += 1
            iterator += 1

        return recommendations

    def make_top_included_recommendations(self, playlist):
        """
        recommend the most playlists-included tracks

        :param playlist:
        :return:
        """
        recommendations = []
        top_included = self.db.get_top_included()
        iterator = 0
        count = 0
        already_included = self.db.get_playlist_tracks(playlist)
        target_tracks = self.db.get_target_tracks()
        while count < 5:
            item = top_included[iterator][0]
            if (item not in already_included) and (item not in recommendations) and (item in target_tracks):
                recommendations.append(item)
                count += 1
            iterator += 1

        return recommendations

    def make_top_tag_recommendations(self, active_playlist, target_tracks, recommendations):
        """
        This method takes into account tags. For the active playlist all tags of the tracks are computed,
        then for every recommendable track the comparison of the tags is used taking into account:

        * the matched tags with respect to the total tags number of the playlists
        * secondly the matched tags over the total tags of the track
        * lastly the position of the track in top included

        :param active_playlist:
        :return:
        """
        #already_included = self.db.get_playlist_user_tracks(active_playlist)

        knn = 5 - len(recommendations)
        active_tracks = self.db.get_playlist_tracks(active_playlist) # get already included tracks
        already_included = active_tracks
        active_tags = []
        [active_tags.extend(self.db.get_track_tags(track)) for track in active_tracks] # get the tags from the active tracks
        active_tags_set = set(active_tags)

        top_tracks = []
        track_playlists_map = self.db.get_track_playlists_map()

        for track in target_tracks: # make the actual recommendation
            track_duration = self.db.get_track_duration(track) # get the track length
            tags = self.db.get_track_tags(track)
            if track not in already_included and (track_duration > 30000 or track_duration < 0) and track not in recommendations:
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
        recommendations = [recommendation[0] for recommendation in top_tracks[0:knn]]
        return recommendations


    def make_tf_idf_recommendations(self, playlist, target_tracks, recommendations):

        possible_recommendations = []
        knn = 5 - len(recommendations)
        playlist_tracks = self.db.get_playlist_tracks(playlist)
        playlist_tracks_set = set(playlist_tracks)
        if playlist_tracks == []:
            raise ValueError("playlist is empty")
        average_playlist_duration = sum([self.db.get_track_duration(track) for track in playlist_tracks])/len(playlist_tracks)

        playlist_features = [tag for track in playlist_tracks for tags in self.db.get_track_tags(track) for tag in tags]
        playlist_features.extend(self.db.get_titles_playlist(playlist))
        playlist_features_set = list(set(playlist_features))
        if len(playlist_features_set) == 0:
            raise ValueError("playlist have no features!")
        tf_idf_playlist = []
        #already_included = self.db.get_playlist_user_tracks(playlist)
        for tag in playlist_features_set:
            tf = playlist_features.count(tag) / float(len(playlist_features))
            idf = self.db.get_tag_idf(tag)
            tf_idf = tf * idf
            tf_idf_playlist.append(tf_idf)

        for track in target_tracks:
            tags = self.db.get_track_tags(track)
            track_duration = self.db.get_track_duration(track)
            if track not in playlist_tracks_set and track not in recommendations and (track_duration > 30000 or track_duration < 0):
                tf_idf_track = []
                for tag in tags:
                    tf = 1.0 /len(tags)
                    idf = self.db.get_tag_idf(tag)
                    tf_idf = tf * idf
                    tf_idf_track.append(tf_idf)

                num_cosine_sim = [tf_idf_track[tags.index(tag)] * tf_idf_playlist[playlist_features_set.index(tag)] for
                                  tag in tags if tag in playlist_features_set]

                den_cosine_sim = math.sqrt(sum([i ** 2 for i in tf_idf_playlist])) * math.sqrt(
                    sum([i ** 2 for i in tf_idf_track]))
                try:
                    cosine_sim = sum(num_cosine_sim) / (den_cosine_sim)
                except ZeroDivisionError:
                    cosine_sim = 0
                possible_recommendations.append([track, cosine_sim])

        possible_recommendations.sort(key=itemgetter(1), reverse=True)
        recommendations.extend([recommendation for recommendation, value in possible_recommendations[0:knn]])
        return recommendations

    def make_artists_recommendations(self, playlist, knn):

        possible_recommendations = []

        playlist_tracks = self.db.get_playlist_tracks(playlist)
        if playlist_tracks == []:
            raise LookupError("The playlist is empty")
        artists_percentages = []
        average_playlist_duration = sum((self.db.get_track_duration(track) for track in playlist_tracks))/len(playlist_tracks)

        for track in playlist_tracks:
            artist_tracks = self.db.get_artist_tracks(track)
            float_is_in_artist_tracks = (float(i) for i in (track in artist_tracks for track in playlist_tracks))
            artist_percentage = sum(float_is_in_artist_tracks)/len(playlist_tracks)
            artist_id = self.db.get_artist(track)
            artists_percentages.append([artist_id, artist_percentage, artist_tracks])

        artists_percentages.sort(key = itemgetter(1),reverse = True)
        most_in_artist = artists_percentages[0][1]

        if most_in_artist > 0.7:
            artist_tracks = artists_percentages[0][2]
        else:
            raise ValueError("The playlist have no dominant artist")

        target_tracks = helper.diff_list(artist_tracks, playlist_tracks)
        if target_tracks == []:
            raise ValueError("The playlist contains all the songs of the artist!")

        playlist_tracks_set = set(playlist_tracks)

        playlist_features = []
        [playlist_features.extend(self.db.get_track_tags(track)) for track in playlist_tracks]
        playlist_features_set = list(set(playlist_features))
        if len(playlist_features_set) == 0:
            raise ValueError("playlist have no features!")
        tf_idf_playlist = []

        for tag in playlist_features_set:
            tf = playlist_features.count(tag) / float(len(playlist_features))
            idf = self.db.get_tag_idf(tag)
            tf_idf = tf * idf
            tf_idf_playlist.append(tf_idf)

        for track in target_tracks:
            tags = self.db.get_track_tags(track)
            track_duration = self.db.get_track_duration(track)
            if (track_duration > 30000 or track_duration < 0):
                tf_idf_track = []
                for tag in tags:
                    tf = 1.0 / len(tags)
                    idf = self.db.get_tag_idf(tag)
                    tf_idf = tf * idf
                    tf_idf_track.append(tf_idf)

                num_cosine_sim = [tf_idf_track[tags.index(tag)] * tf_idf_playlist[playlist_features_set.index(tag)] for
                                  tag in tags if tag in playlist_features_set]

                den_cosine_sim = math.sqrt(sum([i ** 2 for i in tf_idf_playlist])) * math.sqrt(
                    sum([i ** 2 for i in tf_idf_track]))

                try:
                    cosine_sim = sum(num_cosine_sim) / (den_cosine_sim)
                except ZeroDivisionError:
                    cosine_sim = 0
                if cosine_sim > 0.75:
                    possible_recommendations.append([track, cosine_sim])

        possible_recommendations.sort(key=itemgetter(1), reverse=True)
        recommendations = [recommendation for recommendation, value in possible_recommendations[0:knn]]
        return recommendations

    def make_tf_idf_titles_recommendations(self, playlist, target_tracks, knn):

        possible_recommendations = []

        playlist_tracks = self.db.get_playlist_tracks(playlist)
        playlist_tracks_set = set(playlist_tracks)

        playlist_titles = self.db.get_titles_playlist(playlist)
        tf_idf_titles_playlist = []

        tf_idf_titles_playlist = [self.db.get_title_idf(title)/float(len(playlist_titles)) for title in playlist_titles]

        '''
        the above list comprehension is equal to the following code
        for title in playlist_titles:
            tf = 1 / float(len(playlist_titles))
            idf = self.db.get_title_idf(title)
            tf_idf = tf * idf
            tf_idf_titles_playlist.append(tf_idf)
        '''

        for track in target_tracks:
            if track not in playlist_tracks_set and self.db.get_track_duration(track) > 60000:

                titles = self.db.get_titles_track(track)

                tf_idf_title = [self.db.get_title_idf(title)/len(titles) for title in titles]
                '''
                the above list comprehension is equivalent to the following code but (hopefully) a bit faster
                tf_idf_title = []
                for title in titles:
                    tf = 1.0 / len(titles)
                    idf = self.db.get_title_idf(title)
                    tf_idf = tf * idf
                    tf_idf_title.append(tf_idf)
                '''

                num_cosine_sim = [tf_idf_title[titles.index(title)] * tf_idf_titles_playlist[playlist_titles.index(title)] for
                                  title in titles if title in playlist_titles]

                den_cosine_sim = math.sqrt(sum([i ** 2 for i in tf_idf_titles_playlist])) * math.sqrt(
                    sum([i ** 2 for i in tf_idf_title]))
                try:
                    cosine_sim = sum(num_cosine_sim) / den_cosine_sim
                except ZeroDivisionError:
                    cosine_sim = 0

                possible_recommendations.append([track, cosine_sim])

        possible_recommendations.sort(key=itemgetter(1), reverse=True)
        recommendations = [recommendation for recommendation, value in possible_recommendations[0:knn]]
        return recommendations


    def combined_top_tag_tfidf_recommendations(self, playlist):
        """
        this function combines the top tag and the tf idf recommendations
        :return:
        """
        knn = 125
        tracks = self.db.get_target_tracks()
        filtered_tracks = self.make_top_tag_recommendations(playlist, tracks, knn)
        return self.make_tf_idf_recommendations(playlist, filtered_tracks, 5)

    def combined_tfidf_tags_tfidf_titles_recommendations(self, playlist):
        """
        this function combines the top tag and the tf idf recommendations
        :return:
        """
        knn = 50
        tracks = self.db.get_target_tracks()
        filtered_tracks = self.make_tf_idf_recommendations(playlist, tracks, knn)

        return self.make_tf_idf_titles_recommendations(playlist, filtered_tracks, 5)

    def combined_tfidf_top_tag_recommendations(self, playlist):
        """
        this function combines the top tag and the tf idf recommendations
        :return:
        """
        knn = 125
        tracks = self.db.get_target_tracks()
        filtered_tracks = self.make_tf_idf_recommendations(playlist, tracks, knn)

        return self.make_top_tag_recommendations(playlist, filtered_tracks, 5)

    def combined_top_tag_tfidf_titles_recommendations(self, playlist):
        """
        this function combines the top tag and the tf idf recommendations
        :return:
        """
        knn = 350
        tracks = self.db.get_target_tracks()
        filtered_tracks = self.make_top_tag_recommendations(playlist, tracks, knn)
        return self.make_tf_idf_titles_recommendations(playlist, filtered_tracks, knn)

    def combined_tfidf_tfidf_titles_recommendations(self, playlist):
        """
        this function combines the top tag and the tf idf recommendations
        :return:
        """
        knn = 50
        tracks = self.db.get_target_tracks()
        filtered_tracks = self.make_tf_idf_recommendations(playlist, tracks, knn)
        return self.make_tf_idf_titles_recommendations(playlist, filtered_tracks, 5)


    def make_bad_tf_idf_recommendations(self, playlist):
        possible_recommendations = []

        playlist_tracks = self.db.get_playlist_tracks(playlist)
        playlist_tracks_set = set(playlist_tracks)


        playlist_features = []
        [playlist_features.extend(self.db.get_track_tags(track)) for track in playlist_tracks]
        playlist_features_set = list(set(playlist_features))
        tf_idf_playlist = []

        tf_idf_playlist = [self.db.get_tag_idf(tag)*playlist_features.count(tag)/float(len(playlist_features)) for tag in playlist_features_set]
        '''
        the above expression simplify the following
        for tag in playlist_features_set:
            tf = playlist_features.count(tag) / float(len(playlist_features))
            idf = self.db.get_tag_idf(tag)
            tf_idf = tf * idf
            tf_idf_playlist.append(tf_idf)
        '''

        tracks_tags = []
        tf_idf_tracks_tags = []

        for track in playlist_tracks:

            tags = self.db.get_track_tags(track)
            tracks_tags.append(tags)
            tf_idf_s = [self.db.get_tag_idf(tag)/float(len(tags)) for tag in tags]

            '''
            equivalent to the above list comprehension
            tf_idf_s = []
            for tag in tags:
                tf = 1 / float(len(tags))
                idf = self.db.get_tag_idf(tag)
                tf_idf = tf * idf
                tf_idf_s.append(tf_idf)
            '''
            tf_idf_tracks_tags.append(tf_idf_s)


        for track in self.db.get_target_tracks():
            if track not in playlist_tracks_set and self.db.get_track_duration(track)>60000:
                tags = self.db.get_track_tags(track)
                tf_idf_track = [self.db.get_tag_idf(tag)/float(len(tags)) for tag in tags]
                '''
                tf_idf_track = []
                for tag in tags:
                    tf = 1/float(len(tags))
                    idf = self.db.get_tag_idf(tag)
                    tf_idf = tf * idf
                    tf_idf_track.append(tf_idf)
                '''

                mean_cosine_sim = []
                for track_tags in tracks_tags:
                    num_cosine_sim = [tf_idf_track[tags.index(tag)] * tf_idf_tracks_tags[tracks_tags.index(track_tags)][track_tags.index(tag)] for tag in tags if tag in track_tags]

                    '''
                    num_cosine_sim = []
                    for tag in tags:
                        if tag in track_tags:
                            num_cosine_sim.append(tf_idf_track[tags.index(tag)] * tf_idf_tracks_tags[tracks_tags.index(track_tags)][track_tags.index(tag)])
                    '''
                    den_cosine_sim = math.sqrt(sum([i**2 for i in tf_idf_tracks_tags[tracks_tags.index(track_tags)]])) * math.sqrt(sum([i**2 for i in tf_idf_track]))

                    try:
                        cosine_sim = sum(num_cosine_sim)/float(den_cosine_sim)
                    except ZeroDivisionError:
                        cosine_sim = 0
                    mean_cosine_sim.append(cosine_sim)
                try:
                    value = sum(mean_cosine_sim)/float(len(mean_cosine_sim))
                except ZeroDivisionError:
                    value = 0
                possible_recommendations.append([track, value])

        recommendations = sorted(possible_recommendations, key=itemgetter(1), reverse=True)[0:5]
        return [recommendation for recommendation, value in recommendations]
