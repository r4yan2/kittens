from recommender import Recommender
from database import Database
from multiprocessing import Process
import helper

config = {'test': 'True',
         'debug': 'False',
         'filename': 'result',
         'start_from': '0',
          'parallel': 'False'}

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
test = False
parallel = False
db = Database(test)
recommender_system = Recommender(db)
choice = input("Please select one >  ")
if parallel:
    print "\nVROOOOOOOOOMMMMMMMMMMMMMMMMMMMMMMM\nParallel Engine ACTIVATION\n"+"CORE TRIGGERED\n"*2+"VVVRRRRROOOOOOOOOOOOMMMMMMMMMMMMM\n"
    to_recommend = db.get_test_list()
    pieces = 2
    piece = (len(to_recommend)+1) / pieces
    for i in xrange(pieces):
        p = Process(target=recommender_system.run, args=(choice, False, False, "result", i*piece, (i+1)*piece, i, ))
        p.daemon = True
        p.start()
    for i in xrange(pieces):
        p.join()
else:
    recommender_system.run(choice)
