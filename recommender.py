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
                recommendations = self.make_top_tag_recommendations(target, self.db.get_target_tracks(), 5)
            elif choice == 4:
                try:
                    recommendations = self.make_tf_idf_recommendations(target, self.db.get_target_tracks(), 5)
                except ValueError:
                    recommendations = self.make_top_included_recommendations(target)
                if len(recommendations) < 5:
                    recommendations.extend(self.make_top_tag_recommendations(target, self.db.get_target_tracks(), 5-len(recommendations)))
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
                        recommendations = self.make_tf_idf_recommendations(target, self.db.get_target_tracks(), 5)
                    except ValueError: # this may happend when the playlist have 1-2 tracks with no features (fuck it)
                        recommendations = self.make_top_included_recommendations(target)
                if len(recommendations) < 5: # if there are not enough artist tracks to recommend or if the tracks have a strage avg duration
                    recommendations.extend(self.make_tf_idf_recommendations(target, self.db.get_target_tracks(), 5 - len(recommendations)))
                if len(recommendations) < 5:
                    recommendations.extend(self.make_top_tag_recommendations(target, self.db.get_target_tracks(), 5-len(recommendations)))

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

    def make_top_tag_recommendations(self, active_playlist, target_tracks, knn):
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
            if track not in already_included and track_duration > 60000:
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


    def make_tf_idf_recommendations(self, playlist, target_tracks, knn):

        possible_recommendations = []

        playlist_tracks = self.db.get_playlist_tracks(playlist)
        playlist_tracks_set = set(playlist_tracks)
        if playlist_tracks == []:
            raise ValueError("playlist is empty")
        average_playlist_duration = sum([self.db.get_track_duration(track) for track in playlist_tracks])/len(playlist_tracks)

        playlist_features = []
        [playlist_features.extend(self.db.get_track_tags(track)) for track in playlist_tracks]
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
            if track not in playlist_tracks_set and self.db.get_track_duration(track) > 60000 and math.fabs(self.db.get_track_duration(track) - average_playlist_duration) <= 1 * average_playlist_duration:
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
                #if len(playlist_features_set) == 1:
                    #shrink = 1
               # else:
                    #shrink = 1.0/math.log(len(tags), len(playlist_features_set))
                try:
                    cosine_sim = sum(num_cosine_sim) / (den_cosine_sim)
                except ZeroDivisionError:
                    cosine_sim = 0

                possible_recommendations.append([track, cosine_sim])

        possible_recommendations.sort(key=itemgetter(1), reverse=True)
        recommendations = [recommendation for recommendation, value in possible_recommendations[0:knn]]
        return recommendations

    def make_artists_recommendations(self, playlist, knn):

        possible_recommendations = []

        playlist_tracks = self.db.get_playlist_tracks(playlist)
        if playlist_tracks == []:
            raise LookupError("The playlist is empty")
        artists_percentages = []
        average_playlist_duration = sum([self.db.get_track_duration(track) for track in playlist_tracks])/len(playlist_tracks)

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

        tracks_not_in_playlist = helper.diff_list(artist_tracks, playlist_tracks)
        if tracks_not_in_playlist == []:
            raise ValueError("The playlist contains all the songs of the artist")

        playlist_tracks_set = set(playlist_tracks)

        playlist_features = []
        [playlist_features.extend(self.db.get_track_tags(track)) for track in playlist_tracks]
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

        for track in tracks_not_in_playlist:
            tags = self.db.get_track_tags(track)
            track_duration = self.db.get_track_duration(track)
            if track not in playlist_tracks_set and track_duration > 60000 and math.fabs(track_duration - average_playlist_duration) < 0.5 * average_playlist_duration:
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
                #if len(playlist_features_set) == 1:
                    #shrink = 1
               # else:
                    #shrink = 1.0/math.log(len(tags), len(playlist_features_set))
                try:
                    cosine_sim = sum(num_cosine_sim) / (den_cosine_sim)
                except ZeroDivisionError:
                    cosine_sim = 0

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

    def make_top_n_recommendations(self, user, shrink):

        top_n = self.db.get_top_n_w_shrink() if shrink else self.db.get_top_n()
        return top_n

    def make_top_n_feature_aware_recommendations(self, user):
        '''
        Questa raccomendazione dovrebbe tenere in conto le feature presenti in un film, ad esempio
        se un film ha 5 feature che l'utente ha valutato 8 allora la media delle 5 risulta 8. Il problema
        sorge quando non tutte le feature sono state valutate dall'utente, es 5 feature 7 8 9 0 0.
        Bisogna stabilire un parametro per capire se raccomandare o meno quel film, es le feature che contano 0
        vengono scartate o vengono considerate come se valessero 5 o viene stabilita una penalty
        in base al numero di feature valutate sul totale delle feature di un film.
        :param user:
        :return:
        '''
        item_set = self.db.get_item_set()
        for item in item_set:
            features = self.db.get_features(item)
            for feature in features:
                user_evaluation = self.db.get_user_feature_evaluation(user, feature)
                if user_evaluation > 0:
                    pass

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
        features_avg = defaultdict(lambda: 0, {})
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

    def get_kittens_recommendations(self, helper, user):
        user_items = self.db.get_user_evaluations(user)
        similarities = self.db.get_user_similarities(user)
        recommendations = []
        possible_recommendations = []
        for user_iterator in map(lambda x: x[1], similarities):
           user_iterator_items = self.db.get_user_evaluations(user_iterator)
           possible_recommendations.extend(helper.diff_list(user_iterator_items, user_items))
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
