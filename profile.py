from database import Database
from recommender import Recommender
import cProfile
db=Database(1)
rm=Recommender(db)

def to_profile():
    result = []
    for playlist in db.get_target_playlists():
        playlist_tracks = db.get_playlist_tracks(playlist)
        playlist_tracks_set = set(playlist_tracks)

        playlist_features = []
        [playlist_features.extend(db.get_track_tags(track)) for track in playlist_tracks]
        playlist_features_set = list(set(playlist_features))
        tf_idf_playlist = []

        for tag in playlist_features_set:
            tf = playlist_features.count(tag) / float(len(playlist_features))
            idf = db.get_tag_idf(tag)
            tf_idf = tf * idf
            tf_idf_playlist.append(tf_idf)
        result.append(tf_idf_playlist)
    return result

cProfile.run('to_profile')

#cProfile.run('recomm = rm.make_tf_idf_recommendations(3042855, db.get_target_tracks(), 5)')
#cProfile.run('rm.check_recommendations(3042855, recomm)')
#cProfile.run('recomm = rm.make_tf_idf_recommendations(5935981, db.get_target_tracks(), 5)')
#cProfile.run('rm.check_recommendations(5935981, recomm)')
#cProfile.run('recomm = rm.combined_top_tag_tfidf_recommendations(4740724)')
#cProfile.run('rm.check_recommendations(4740724, recomm)')
#cProfile.run('recomm = rm.combined_tfidf_top_tag_recommendations(5935981)')
#cProfile.run('rm.check_recommendations(5935981, recomm)')
#cProfile.run('recomm = rm.combined_tfidf_top_tag_recommendations(4740724)')
#cProfile.run('rm.check_recommendations(4740724, recomm)')
