from recommender import Recommender
from database import Database

config = {'test': 'True',
         'debug': 'False',
         'filename': 'result',
         'start_from': '0'}

disclaimer = """
    --> Kitt<3ns Recommendation ENGINE <--

    USAGE:

    [0] Random
    [1] Top N
    [2] Top N with shrink factor
    [3] Top Viewed

    Please wait until the Engine is ready, then select your choice
    """
print disclaimer
test = True
db = Database(test)
recommender_system = Recommender(db, test)
choice = input("Please select one >  ")
recommender_system.run(choice)