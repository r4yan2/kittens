from database import Database

db = Database(0, tags="tags")

tracks_map = db.get_tracks_map()

tags = [tag for line in tracks_map.values() for tag in line[4]]
tags_set = set(tags)
tags_sorted = sorted(tags_set)

tags_joined = ','.join([str(elem) for elem in tags_sorted])

file_to = open("data/tag_encoding", "wb+")
file_to.write(tags_joined)

file_to.close()
