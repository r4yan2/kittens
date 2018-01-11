from multiprocessing import Process, Queue, Manager
from database import Database
from collections import Counter
import csv
import gc
from operator import itemgetter
import time
import sys
import helper

def compute_item_item_similarities(db, q_in, q_out, number):
    gc.collect()
    artists = sorted(set([db.get_artist(track) for track in db.get_tracks()]))
    while True:
        (identifier, i) = q_in.get()
        if i == -1:
            time.sleep(30)
            break

        i_stuffs = db.get_tag_playlists(i)
        if i_stuffs == []:
            q_out.put(-1)
            continue

        similarities = [[i,j,helper.jaccard(i_stuffs, j_stuffs)] for j in artists for j_stuffs in [db.get_tag_playlists(j)] if j_stuffs != [] and i < j]

        similarities = [[i,j,v] for i,j,v in similarities if v > 0]

        if similarities == []:
            q_out.put(-1)
            continue
        else:
            q_out.put(sorted(similarities, key=itemgetter(2), reverse=True)[0:150])

fp = open('data/artist-artist-similarities0.csv', 'w', 0)
writer = csv.writer(fp, delimiter=',', quoting=csv.QUOTE_NONE)

core=int(sys.argv[1])

db = Database(0)
target_artists =sorted(set([db.get_artist(track) for track in db.get_target_tracks()]))

# Queue(s) for the process, one for input data to the process, the other for the output data to the main process
q_in = Queue()
q_out = Queue()
[q_in.put((i, x)) for i, x in enumerate(target_artists)]

[q_in.put((-1, -1)) for _ in xrange(core)]

proc = [Process(target=compute_item_item_similarities, args=(db, q_in, q_out, i))
        for i in xrange(core)]
for p in proc:
    p.daemon = True
    p.start()

works = len(target_artists)

done = set()

for i in xrange(works):
    r = q_out.get()
    if r == -1:
        continue
    writer.writerows(r)
[p.join() for p in proc]

fp.close()
