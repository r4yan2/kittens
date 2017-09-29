import csv
import sys


class Helper:
    def __init__(self):
        pass

    @staticmethod
    def diff_list(a, b):
        b = set(b)
        return [aa for aa in a if aa not in b]

    @staticmethod
    def write(filename, content):
        writer = (csv.writer(open('data/'+filename, 'w'), delimiter=','))
        writer.writerows(content)

    @staticmethod
    def tick(completion):
        sys.stdout.write("\r%f%%" % completion)
        sys.stdout.flush()
