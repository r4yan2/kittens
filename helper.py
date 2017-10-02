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
    """
    what should be an existing csv, return the csv reader to be listed
    :param what:
    :return:
    """
    return csv.reader(open('data/'+what+'.csv', 'rb'), delimiter='\t')
