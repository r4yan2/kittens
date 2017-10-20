from database import Database
from recommender import Recommender
import cProfile
db=Database(1)
rm=Recommender(db)

cProfile.run('recomm = rm.make_tf_idf_recommendations(3042855, db.get_target_tracks(), [])')
cProfile.run('rm.check_recommendations(3042855, recomm)')
cProfile.run('recomm = rm.make_tf_idf_recommendations(5935981, db.get_target_tracks(), [])')
cProfile.run('rm.check_recommendations(5935981, recomm)')
