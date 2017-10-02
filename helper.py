import csv
import sys


class Helper:
    def __init__(self, filename, header=None):
        self.fp = open('data/' + filename + '.csv', 'w', 0)
        if not header == None:
            self.writer = csv.writer(self.fp, delimiter=',', quoting=csv.QUOTE_NONE)
            self.writer.writerow(header)

    @staticmethod
    def diff_list(a, b):
        b = set(b)
        return [aa for aa in a if aa not in b]

    def write(self, content):
        try:
            self.writer.writerows(content)
        except AttributeError:
            self.writer = csv.writer(self.fp, delimiter=',', quoting=csv.QUOTE_NONE)
            self.writer.writerows(content)

    def close(self):
        self.fp.close()

    @staticmethod
    def tick(completion):
        sys.stdout.write("\r%f%%" % completion)
        sys.stdout.flush()

def read(what):
    return csv.reader(open('data/'+what+'.csv', 'rb'), delimiter='\t')
    '''
    self.train_csv = csv.reader(open('data/train_final.csv', 'rb'), delimiter='\t')
    self.playlists_csv = csv.reader(open('data/playlists_final.csv', 'rb'), delimiter='\t')
    self.tracks_csv = csv.reader(open('data/tracks_final.csv', 'rb'), delimiter='\t')
    self.target_playlists_csv = csv.reader(open('data/target_playlists.csv', 'rb'), delimiter='\t')
    self.target_tracks_csv = csv.reader(open('data/target_tracks.csv', 'rb'), delimiter='\t')
'''