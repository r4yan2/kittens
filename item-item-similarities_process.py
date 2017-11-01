from multiprocessing import Process, Queue, Manager
from database import Database
from collections import Counter
import csv
import gc
from operator import itemgetter

def compute_item_item_similarities(db, q_in, q_out):
    gc.collect()
    while True:
        (identifier, i) = q_in.get()
        if i == -1:
            break
        playlists = db.get_playlists()

        duration = db.get_track_duration(i)
        if not (duration > 30000 or duration < 0):
            continue

        similarities = []
        scanned_users = set()
        scanned_tracks = set()
        for playlist in playlists:
            user = db.get_playlist_user(playlist)
            if user in scanned_users:
                continue
            user_tracks = db.get_playlist_user_tracks(playlist)
            user_tracks_counter = Counter(user_tracks)
            user_tracks_set = set(user_tracks)

            scanned_users.add(user)
            numerator = []
            denominator = []
            for j in user_tracks_set:
                if j in scanned_tracks:
                    continue
                scanned_tracks.add(j)

                numerator.append(user_tracks_counter[i] * user_tracks_counter[j])

                denominator.append(user_tracks_counter[i] * user_tracks_counter[i] + user_tracks_counter[j] * user_tracks_counter[j] -
                                   user_tracks_counter[i] * user_tracks_counter[j])
            try:
                similarity = sum(numerator) / float(sum(denominator))
            except ZeroDivisionError:
                continue

            similarities.append((i,j,similarity))
        q_out.put(sorted(similarities, key=itemgetter(2), reverse=True)[0:500])

fp = open('data/item-item-similarities.csv', 'w', 0)
writer = csv.writer(fp, delimiter=',', quoting=csv.QUOTE_NONE)

core=4

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

proc = [Process(target=compute_item_item_similarities, args=(db, q_in, q_out))
        for i in xrange(core)]
for p in proc:
    p.daemon = True
    p.start()

for i in xrange(len(target_tracks)):
    r = q_out.get()
    writer.writerows(r)

[p.join() for p in proc]

fp.close()
