from database import Database
from recommender import Recommender
import cProfile
db=Database(0)
rm=Recommender(db)

cProfile.run('recomm = rm.make_tf_idf_recommendations(3042855, db.get_target_tracks(), 5)')
cProfile.run('rm.check_recommendations(3042855, recomm)')
cProfile.run('recomm = rm.make_tf_idf_recommendations(5935981, db.get_target_tracks(), 5)')
cProfile.run('rm.check_recommendations(5935981, recomm)')
cProfile.run('recomm = rm.combined_top_tag_tfidf_recommendations(4740724)')
cProfile.run('rm.check_recommendations(4740724, recomm)')
cProfile.run('recomm = rm.combined_tfidf_top_tag_recommendations(5935981)')
cProfile.run('rm.check_recommendations(5935981, recomm)')
cProfile.run('recomm = rm.combined_tfidf_top_tag_recommendations(4740724)')
cProfile.run('rm.check_recommendations(4740724, recomm)')
