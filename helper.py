import csv


def write(filename, content):
    """
    write the result on the file
    :param content:
    :return:
    """
    fp = open('data/' + filename + '.csv', 'w', 0)
    writer = csv.writer(fp, delimiter=',', quoting=csv.QUOTE_NONE)
    if filename == "result":
        header = ["playlist_id", "track_ids"]
        writer.writerow(header)
    if len(content) > 1:
        writer.writerows(content)
    else:
        writer.writerow(content)
    fp.close()

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
    make the difference between list of lists
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

def cumulative_sum(lis):

    total = 0
    for x in lis:
        total += x
        yield total

def multiply_lists(lst1, lst2):

    return [a * b for a, b in zip(lst1, lst2)]

def divide_lists(lst1, lst2):

    return [a / b for a, b in zip(lst1, lst2)]

