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
                except ValueError:
                    recommendations = self.make_top_included_recommendations(target)
                knn = len(recommendations)
                if knn < 5:
                    recommendations = recommendations + self.make_top_tag_recommendations(target, recommendations=recommendations)
            elif choice == 5:
                recommendations = self.combined_top_tag_tfidf_recommendations(target)
            elif choice == 6:
                recommendations = self.combined_tfidf_top_tag_recommendations(target)
            elif choice == 7:
                recommendations = self.make_tf_idf_titles_recommendations(target)
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
                    except ValueError: # this may happend when the playlist have 1-2 tracks with no features (fuck it)
                        recommendations = self.make_top_included_recommendations(target)
                if len(recommendations) < 5: # if there are not enough artist tracks to recommend or if the tracks have a strage avg duration
                    recommendations = recommendations + self.make_tf_idf_recommendations(target, recommendations=recommendations)
                if len(recommendations) < 5:
                    recommendations = recommendations + self.make_top_tag_recommendations(target, recommendations=recommendations)

            # doing testing things if test mode enabled
            if test:
                test_result = self.check_recommendations(target, recommendations)
                q_out.put([test_result, number])

            else:
                # else put the result into the out queue
                q_out.put([identifier, target, recommendations])

    def make_random_recommendations(self, playlist, target_tracks=[], recommendations=[], knn=5):
        """
        take 5 random tracks and recommend them

        :param playlist:
        :return:
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
        recommend the top listened tracks

        :param playlist:
        :return:
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
        recommend the most playlists-included tracks

        :param playlist:
        :return:
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

        * the matched tags with respect to the total tags number of the playlists
        * secondly the matched tags over the total tags of the track
        * lastly the position of the track in top included

        :param active_playlist:
        :return:
        """
        #already_included = self.db.get_playlist_user_tracks(active_playlist)

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
        if target_tracks == []:
            target_tracks = self.db.get_target_tracks()

        knn = knn - len(recommendations)



        possible_recommendations = []
        playlist_tracks = self.db.get_playlist_tracks(playlist)
        playlist_tracks_set = set(playlist_tracks)
        if playlist_tracks == []:
            print "playlist empty"
            raise ValueError("playlist is empty")


        playlist_features = [tag for track in playlist_tracks for tag in self.db.get_track_tags(track)] #+ \
                            #[item * (-(10**10)) for track in playlist_tracks for item in self.db.get_titles_track(track)]

        playlist_features_set = list(set(playlist_features))
        if len(playlist_features_set) == 0:
            print "playlist have no feature"
            raise ValueError("playlist have no features!")

        tf_idf_playlist = [(1.0 + math.log(playlist_features.count(tag), 10)) * self.db.get_tag_idf(tag)
                           for tag in playlist_features_set]

        """
        above list comprehension summarize the following:
        
        tf_idf_playlist = []
        for tag in playlist_features_set:
            tf = 1.0 + math.log(playlist_features.count(tag), 10)
            idf = self.db.get_tag_idf(tag)
            tf_idf = tf * idf
            tf_idf_playlist.append(tf_idf)
        """
        for track in target_tracks:
            tags = self.db.get_track_tags(track)  #+ [item * (-(10**10)) for item in self.db.get_titles_track(track)]
            track_duration = self.db.get_track_duration(track)
            if track not in playlist_tracks_set and (track_duration > 30000 or track_duration < 0) and track not in recommendations:

                tf_idf_track = [1.0 * self.db.get_tag_idf(tag) for tag in tags]

                """
                tf_idf_track = []
                for tag in tags:
                    tf = 1.0
                    idf = self.db.get_tag_idf(tag)
                    tf_idf = tf * idf
                    tf_idf_track.append(tf_idf)
                """

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
        recs = recommendations + [recommendation for recommendation, value in possible_recommendations[0:knn]]
        return recs

    def make_artists_recommendations(self, playlist, recommendations=[], knn=5):

        knn -= len(recommendations)

        possible_recommendations = []

        playlist_tracks = self.db.get_playlist_tracks(playlist)
        if playlist_tracks == []:
            raise LookupError("The playlist is empty")

        artists_percentages = []
        for track in playlist_tracks:
            artist_tracks = self.db.get_artist_tracks(track)
            float_is_in_artist_tracks = map(float, [track in artist_tracks for track in playlist_tracks])
            artist_percentage = sum(float_is_in_artist_tracks)/len(playlist_tracks)
            artist_id = self.db.get_artist(track)
            artists_percentages.append([artist_id, artist_percentage, artist_tracks])

        artists_percentages.sort(key = itemgetter(1),reverse = True)
        most_in_artist = artists_percentages[0][1]

        if most_in_artist > 0.75:
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

        """
        summed-up by the above list-comprehension
        tf_idf_playlist = []
        for tag in playlist_features_set:
            tf = 1.0 + math.log(playlist_features.count(tag))
            idf = self.db.get_tag_idf(tag)
            tf_idf = tf * idf
            tf_idf_playlist.append(tf_idf)
        """

        for track in target_tracks:
            tags = self.db.get_track_tags(track)
            track_duration = self.db.get_track_duration(track)
            if (track_duration > 30000 or track_duration < 0):
                tf_idf_track = []
                for tag in tags:
                    tf = 1.0
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
                if cosine_sim > 0.8:
                    possible_recommendations.append([track, cosine_sim])

        possible_recommendations.sort(key=itemgetter(1), reverse=True)
        recs = recommendations + [recommendation for recommendation, value in possible_recommendations[0:knn]]
        return recs

    def make_tf_idf_titles_recommendations(self, playlist, target_tracks=[], recommendations=[], knn=5):

        if target_tracks == []:
            target_tracks = self.db.get_target_tracks()

        knn -= len(recommendations)

        possible_recommendations = []

        playlist_tracks = self.db.get_playlist_tracks(playlist)
        playlist_tracks_set = set(playlist_tracks)

        playlist_titles = self.db.get_titles_playlist(playlist)

        if playlist_titles == []:
            raise ValueError("no titles!")

        tf_idf_titles_playlist = [(1.0 + math.log(len(playlist_titles), 10)) * self.db.get_title_idf(title) for title in playlist_titles]

        '''
        the above list comprehension is equal to the following code
        for title in playlist_titles:
            tf = float(len(playlist_titles))
            idf = self.db.get_title_idf(title)
            tf_idf = tf * idf
            tf_idf_titles_playlist.append(tf_idf)
        '''

        for track in target_tracks:
            track_duration = self.db.get_track_duration(track)
            if track not in playlist_tracks_set and (track_duration > 30000 or track_duration < 0):

                titles = self.db.get_titles_track(track)

                tf_idf_title = [self.db.get_title_idf(title) for title in titles]
                '''
                the above list comprehension is equivalent to the following code but (hopefully) a bit faster
                tf_idf_title = []
                for title in titles:
                    tf = 1.0
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
        return recommendations + [recommendation for recommendation, value in possible_recommendations[0:knn]]


    def combined_top_tag_tfidf_recommendations(self, playlist):
        """
        this function combines the top tag and the tf idf recommendations
        :return:
        """
        knn = 125
        filtered_tracks = self.make_top_tag_recommendations(playlist, knn=knn)
        return self.make_tf_idf_recommendations(playlist, target_tracks=filtered_tracks)

    def combined_tfidf_tags_tfidf_titles_recommendations(self, playlist):
        """
        this function combines the top tag and the tf idf recommendations
        :return:
        """
        knn = 50
        filtered_tracks = self.make_tf_idf_recommendations(playlist, knn=knn)
        return self.make_tf_idf_titles_recommendations(playlist, target_tracks=filtered_tracks)

    def combined_tfidf_top_tag_recommendations(self, playlist):
        """
        this function combines the top tag and the tf idf recommendations
        :return:
        """
        knn = 125
        filtered_tracks = self.make_tf_idf_recommendations(playlist, knn=knn)
        return self.make_top_tag_recommendations(playlist, target_tracks=filtered_tracks)

    def combined_top_tag_tfidf_titles_recommendations(self, playlist):
        """
        this function combines the top tag and the tf idf recommendations
        :return:
        """
        knn = 350
        filtered_tracks = self.make_top_tag_recommendations(playlist, knn=knn)
        return self.make_tf_idf_titles_recommendations(playlist, target_tracks=filtered_tracks)

    def combined_tfidf_tfidf_titles_recommendations(self, playlist):
        """
        this function combines the top tag and the tf idf recommendations
        :return:
        """
        knn = 100
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

        if target_tracks == []:
            target_tracks = self.db.get_target_tracks()

        knn -= len(recommendations)
        possible_recommendations = []
        logging.debug("playlist %s" % playlist)
        playlist_tracks = self.db.get_playlist_tracks(playlist)
        playlist_tracks_set = set(playlist_tracks)
        if playlist_tracks == []:
            print "playlist empty"
            raise ValueError("playlist is empty")
        logging.debug("playlist tracks: %s" % playlist_tracks)
        logging.debug("len playlist tracks: %i" % len(playlist_tracks))
        normalized_rating = [self.db.get_global_effect(track) for track in playlist_tracks]
        logging.debug("normalized ratings: %s" % normalized_rating)
        logging.debug("len normalized ratings: %i" % len(normalized_rating))
        tracks_tags = []
        tf_idf_tracks_tags = []

        for track in playlist_tracks:

            tags = self.db.get_track_tags(track)
            tracks_tags.append(tags)
            tf_idf_s = [self.db.get_tag_idf(tag) for tag in tags]

            '''
            equivalent to the above list comprehension
            tf_idf_s = []
            for tag in tags:
                tf = 1.0
                idf = self.db.get_tag_idf(tag)
                tf_idf = tf * idf
                tf_idf_s.append(tf_idf)
            '''
            tf_idf_tracks_tags.append(tf_idf_s)
        logging.debug("tracks tags: %s" % tracks_tags)
        logging.debug("len tracks tags: %i" % len(tracks_tags))
        logging.debug("tf_idf_tracks_tags %s" % tf_idf_tracks_tags)
        logging.debug("len tf_idf_tracks_tags %i" % len(tf_idf_tracks_tags))
        if max([len(track_tags) for track_tags in tracks_tags]) == 0:
            raise ValueError("tracks have no feature")
        logging.debug("entering target_tracks loop")
        for track in target_tracks:
            logging.debug("track %i" % track)
            track_duration = self.db.get_track_duration(track)
            if track not in playlist_tracks_set and (track_duration > 30000 or track_duration < 0) and track not in recommendations:
                tags = self.db.get_track_tags(track)
                if tags == []:
                    continue
                tf_idf_track = [self.db.get_tag_idf(tag) for tag in tags]
                '''
                tf_idf_track = []
                for tag in tags:
                    tf = 1.0
                    idf = self.db.get_tag_idf(tag)
                    tf_idf = tf * idf
                    tf_idf_track.append(tf_idf)
                '''
                logging.debug("tf_idf_track %s" % tf_idf_track)
                logging.debug("len tf_idf_track %i" % len(tf_idf_track))
                denominator = []
                numerator = []
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
                    numerator.append(normalized_rating[tracks_tags.index(track_tags)] * cosine_sim)
                    denominator.append(cosine_sim)
                logging.debug("numerator %s" % numerator)
                logging.debug("len numerator %i" % len(numerator))
                logging.debug("denominator %s" % denominator)
                logging.debug("len denominator %i" % len(denominator))
                try:
                    value = sum(numerator)/sum(denominator)
                except ZeroDivisionError:
                    value = 0
                logging.debug("track,value pair: %s" % [track, value])
                possible_recommendations.append([track, value])

        return recommendations + [recommendation for recommendation, value in sorted(possible_recommendations, key=itemgetter(1), reverse=True)[0:knn]]
