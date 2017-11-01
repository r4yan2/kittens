from database import Database
from collections import Counter
import csv
from Queue import Queue
from threading import Thread
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

        similarities = {}
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
                    similarity = 0

                similarities[(i,j)] = similarity
        q_out.put(sorted([[keys[0],keys[1],value] for keys, value in similarities.items() if value != 0], key=itemgetter(1), reverse=True)[0:150])



fp = open('data/item-item-similarities.csv', 'w', 0)
writer = csv.writer(fp, delimiter=',', quoting=csv.QUOTE_NONE)

core=2

db = Database(1)

target_tracks = db.get_target_tracks()

# Queue(s) for the process, one for input data to the process, the other for the output data to the main process
q_in = Queue()
q_out = Queue()
[q_in.put((i, x)) for i, x in enumerate(target_tracks)]

for p in xrange(core):
    worker = Thread(target=compute_item_item_similarities, args=(db, q_in, q_out))
    worker.setDaemon(True)
    worker.start()

for i in xrange(len(target_tracks)):
    r = q_out.get()
    writer.writerows(r)

fp.close()