from multiprocessing import Process, Queue, Manager
from database import Database
from collections import Counter
import csv
import gc
from operator import itemgetter
import time
import sys
import helper
from collections import Counter

def compute_user_user_similarities(db, q_in, q_out, number):
    gc.collect()
    users = sorted(db.get_users())
    while True:
        (identifier, i) = q_in.get()
        if i == -1:
            time.sleep(30)
            break

        i_stuffs = db.get_user_tracks(i)
        if i_stuffs == []:
            q_out.put(-1)
            continue

        similarities = [[i,j,helper.jaccard(i_stuffs, j_stuffs)] for j in users for j_stuffs in [db.get_user_tracks(j)] if j_stuffs != [] and i < j]

        similarities = [[i,j,v] for i,j,v in similarities if v > 0]

        if similarities == []:
            q_out.put(-1)
            continue
        else:
            q_out.put(sorted(similarities, key=itemgetter(2), reverse=True)[0:200])

fp = open('data/user-user-similarities4.csv', 'w', 0)
writer = csv.writer(fp, delimiter=',', quoting=csv.QUOTE_NONE)

core=int(sys.argv[1])

db = Database(4)
target_users =sorted(db.get_users())

# Queue(s) for the process, one for input data to the process, the other for the output data to the main process
q_in = Queue()
q_out = Queue()
[q_in.put((i, x)) for i, x in enumerate(target_users)]

[q_in.put((-1, -1)) for _ in xrange(core)]

proc = [Process(target=compute_user_user_similarities, args=(db, q_in, q_out, i))
        for i in xrange(core)]
for p in proc:
    p.daemon = True
    p.start()

works = len(target_users)

done = set()

for i in xrange(works):
    r = q_out.get()
    if r == -1:
        continue
    writer.writerows(r)
[p.join() for p in proc]

fp.close()
