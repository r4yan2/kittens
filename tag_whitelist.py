from database import Database
from collections import Counter
from operator import itemgetter
import sys

try:
    choice = int(sys.argv[1])
except IndexError:
    choice = int(input("Please insert the choice of database set to open (0-6):\t"))
    
db = Database(choice, tag_whitelist=False)

tracks_map = db.get_tracks_map()

tags = [tag for line in tracks_map.values() for tag in line[4]]
tags_count = Counter(tags)
white=[tag for tag, value in tags_count.items() if value < 300 and value > 1]
print "tag whitelist size", len(white)
tags_joined = ','.join([str(elem) for elem in white])

file_to = open("data/tag_whitelist", "wb+")
file_to.write(tags_joined)

file_to.close()
