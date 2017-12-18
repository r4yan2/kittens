from database import Database
from collections import Counter
from operator import itemgetter

db = Database(0, whitelist=False)

tracks_map = db.get_tracks_map()

tags = [tag for line in tracks_map.values() for tag in line[4]]
tags_count = Counter(tags)
white=[tag for tag, value in tags_count.items() if value < 85 and value > 1]

print "whitelist size", len(white)
tags_joined = ','.join([str(elem) for elem in white])

file_to = open("data/whitelist", "wb+")
file_to.write(tags_joined)

file_to.close()
