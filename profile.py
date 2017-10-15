from database import Database
from recommender import Recommender
import cProfile
db=Database(1)
rm=Recommender(db)
cProfile.run('rm.combined_top_tag_tfidf_recommendations(3042855)')
cProfile.run('rm.combined_top_tag_tfidf_recommendations(5935981)')
cProfile.run('rm.combined_top_tag_tfidf_recommendations(4740724)')
