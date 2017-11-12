from multiprocessing import Process, Queue, Manager
from database import Database
from collections import Counter
import csv
import gc
from operator import itemgetter
import sys

def compute_item_item_similarities(db, q_in, q_out, number):
    gc.collect()
    tracks = set(db.get_tracks())
    playlists = db.get_playlists()
    len_playlists = float(len(playlists))

    while True:
        (identifier, i) = q_in.get()
        print "worker", number, "IN", i
        if i == -1:
            sys.exit(0)
        i_playlists = set(db.get_track_playlists(i))
        if i_playlists == []:
            continue

        duration = db.get_track_duration(i)
        if not (duration > 30000 or duration < 0):
            continue

        # numerator jaccard = A intersection B
        # denominator jaccard = A union B
        # MSE numerator = disjoint element from A and B
        # MSE denominator = A union B

        similarities = [[i, j, ( len(i_playlists.intersection(j_playlists)) / ( len(i_playlists) * ((len(j_playlists)/len_playlists)**0.5)  ) ) * (1.0 - (len(i_playlists.union(j_playlists).difference(i_playlists.intersection(j_playlists)))/float(len(i_playlists.union(j_playlists)))))] for j in tracks for j_playlists in [set(db.get_track_playlists(j))] if j_playlists != [] and i != j]

        q_out.put(sorted(similarities, key=itemgetter(2), reverse=True)[0:30])
        print "worker", number, "OUT", i

fp = open('data/item-item-similarities_prob+mse1.csv', 'w', 0)
writer = csv.writer(fp, delimiter=',', quoting=csv.QUOTE_NONE)

core=3

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

fp.close()
