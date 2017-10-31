from database import Database
import math
db=Database(1)

score = []
for playlist in db.get_target_playlists():
    user_based_collaborative = set(db.get_user_based_collaborative_filtering(playlist))
    relevant_tracks = db.get_playlist_relevant_tracks(playlist)
    user_based_collaborative_score = sum([1 for track in relevant_tracks if track in user_based_collaborative]) / float(len(relevant_tracks))

    collaborative_playlist_similarity_jac = set([track for playlist in db.compute_collaborative_playlists_similarity(playlist, coefficient="jaccard") for track in db.get_playlist_tracks(playlist)])
    collaborative_playlist_similarity_jac_score = sum([1 for track in relevant_tracks if track in collaborative_playlist_similarity_jac]) / float(len(relevant_tracks))

    collaborative_playlist_similarity_cosine = set(
        [track for playlist in db.compute_collaborative_playlists_similarity(playlist, coefficient="cosine") for track
         in db.get_playlist_tracks(playlist)])
    collaborative_playlist_similarity_cosine_score = sum(
        [1 for track in relevant_tracks if track in collaborative_playlist_similarity_cosine]) / float(
        len(relevant_tracks))

    collaborative_playlist_similarity_pea = set(
        [track for playlist in db.compute_collaborative_playlists_similarity(playlist, coefficient="pearson") for track
         in db.get_playlist_tracks(playlist)])
    collaborative_playlist_similarity_pea_score = sum(
        [1 for track in relevant_tracks if track in collaborative_playlist_similarity_pea]) / float(
        len(relevant_tracks))

    collaborative_playlist_similarity_map = set(
        [track for playlist in db.compute_collaborative_playlists_similarity(playlist, coefficient="map") for track
         in db.get_playlist_tracks(playlist)])
    collaborative_playlist_similarity_map_score = sum(
        [1 for track in relevant_tracks if track in collaborative_playlist_similarity_map]) / float(
        len(relevant_tracks))

    score.append([user_based_collaborative_score, collaborative_playlist_similarity_jac_score, collaborative_playlist_similarity_cosine_score, collaborative_playlist_similarity_pea_score, collaborative_playlist_similarity_map_score])

print "AVG score for user_based_collaborative:", sum([a for a, b, c, d, e in score])/len(score)
print "AVG score for collaborative playlist similarity jacard", sum([b for a, b, c, d, e in score])/len(score)
print "AVG score for collaborative playlist similarity cosine", sum([c for a, b, c, d, e in score])/len(score)
print "AVG score for collaborative playlist similarity pearson", sum([d for a, b, c, d, e in score])/len(score)
print "AVG score for collaborative playlist similarity map", sum([e for a, b, c, d, e in score])/len(score)
