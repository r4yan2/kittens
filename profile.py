from database import Database
from recommender import Recommender
import cProfile
db=Database(1)
rm=Recommender(db)

cProfile.run('db.compute_playlists_similarity(3042855)')
#cProfile.run('rm.check_recommendations(3042855, recomm)')
cProfile.run('db.compute_playlists_similarity(5935981)')
#cProfile.run('rm.check_recommendations(5935981, recomm)')
