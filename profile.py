from database import Database
from recommender import Recommender
db=Database(False)
rm=Recommender(db)
rm.make_top_tag_recommendations(8829360)
