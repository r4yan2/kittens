from database import Database
import helper

db=Database(1)
to_write = []
playlists = list(db.get_playlists())
for playlist_a in playlists:
    try:
        to_write.append([playlist_a, db.compute_playlists_similarity(playlist_a)])
    except ValueError:
        pass
    print playlists.index(playlist_a)/float(len(playlists))

helper.write("neighborhood_set_0", to_write, '\t')
    