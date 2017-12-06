from database import Database
from recommender import Recommender
import cProfile
db = Database(1)
rm = Recommender(db)
db.init_item_similarities_epoch()
playlists = db.get_playlists(iterator=False)
tracks = db.get_tracks(iterator=False)
cProfile.run("rm.epoch_iteration(5000, playlists, tracks)")
