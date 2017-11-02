from multiprocessing import Process, Queue, Manager
from database import Database
from collections import Counter
import csv
import gc
from operator import itemgetter

def compute_item_item_similarities(db, q_in, q_out, number):
    gc.collect()
    while True:
        (identifier, i) = q_in.get()
        print "worker", number, "IN", i
        if i == -1:
            break
        playlists = db.get_playlists()
        i_playlists = set(db.get_track_playlists(i))
        if i_playlists == []:
            continue

        duration = db.get_track_duration(i)
        if not (duration > 30000 or duration < 0):
            continue

        similarities = [[i,j,(len(i_playlists.intersection(j_playlists)) / float(len(i_playlists.union(j_playlists)))) * (1.0 - (len(i_playlists.union(j_playlists).difference(i_playlists.intersection(j_playlists)))/float(len(i_playlists.union(j_playlists)))))] for j in db.get_target_playlists_tracks_set() for j_playlists in [set(db.get_track_playlists(j))] if i !=j and j_playlists != []]

        q_out.put(sorted(similarities, key=itemgetter(2), reverse=True)[0:150])
        print "worker", number, "OUT", i

fp = open('data/item-item-similarities.csv', 'w', 0)
writer = csv.writer(fp, delimiter=',', quoting=csv.QUOTE_NONE)

core=1

db = Database(1)

manager = Manager()
ns = manager.Namespace()
ns.db = db
target_tracks = db.get_target_tracks()

# Queue(s) for the process, one for input data to the process, the other for the output data to the main process
q_in = Queue()
q_out = Queue()
[q_in.put((i, x)) for i, x in enumerate(target_tracks)]

[q_in.put((-1, -1)) for _ in xrange(core)]

proc = [Process(target=compute_item_item_similarities, args=(db, q_in, q_out, i))
        for i in xrange(core)]
for p in proc:
    p.daemon = True
    p.start()

for i in xrange(len(target_tracks)):
    r = q_out.get()
    writer.writerows(r)
print "finished working"
[p.join() for p in proc]

fp.close()
