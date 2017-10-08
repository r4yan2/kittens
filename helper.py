import csv
import sys

class Helper:
    def __init__(self, filename, header=None):
        """
        when the helper is initialized it open the file for the result and write the header,
        this should be changed is something less hacky
        
        :param filename: 
        :param header: 
        """
        self.fp = open('data/' + filename + '.csv', 'w', 0)
        if not header == None:
            self.writer = csv.writer(self.fp, delimiter=',', quoting=csv.QUOTE_NONE)
            self.writer.writerow(header)

    def write(self, content):
        """
        write the result on the file
        :param content:
        :return:
        """
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

def diff_list(a, b):
    """
    make the difference of the two lists
    :param a:
    :param b:
    :return:
    """
    b = set(b)
    return [aa for aa in a if aa not in b]

def diff_test_set(a, b):
    """
    pordanna
    :param a:
    :param b:
    :return:
    """
    b = set([str(x[0])+str(x[1]) for x in b])
    return [x for x in a if str(x[0])+str(x[1]) not in b]

def read(what):
    """
    what should be an existing csv, return the csv reader to be listed
    :param what:
    :return:
    """
    return csv.reader(open('data/'+what+'.csv', 'rb'), delimiter='\t')
