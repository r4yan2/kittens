from database import Database
from recommender import Recommender
import cProfile
db=Database()
rm=Recommender()
rm.db=db
cProfile.run('rm.make_tf_idf_recommendations(8829360)')
