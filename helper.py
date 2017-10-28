import csv
import math


def write(filename, content, delimiter_char=','):
    """
    write the result on the file
    :param content:
    :return:
    """
    fp = open('data/' + filename + '.csv', 'w', 0)
    writer = csv.writer(fp, delimiter=delimiter_char, quoting=csv.QUOTE_NONE)
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


def square_sum (x):
    """
    Squares each element of the list and returns the sum
    :param x:
    :return:
    """
    return sum([elem*elem for elem in x])


def pearsonr(x, y):
    """
    return the pearson correlation between x and y
    :param x:
    :param y:
    :return:
    """
    length_x = len(x)
    length_y = len(y)
    if length_x != length_y:
        raise ValueError("The two arrays have different lenght")
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    normalized_x = [(elem - mean_x) for elem in x]
    normalized_y = [(elem - mean_y) for elem in y]

    numerator = sum(multiply_lists(normalized_x, normalized_y))
    denominator = math.sqrt(square_sum(normalized_x) * square_sum(normalized_y))

    return  numerator / denominator

def LevenshteinDistance(s, s_len, t, t_len):
    cost = 0
    if t_len == 0:
        return s_len
    elif s_len == 0:
        return t_len
    if (s[s_len-1] == t[t_len-1]):
        cost = 0
    else:
        cost = 1
    return min(LevenshteinDistance(s, s_len - 1, t, t_len) + 1,
                LevenshteinDistance(s, s_len, t, t_len - 1) + 1,
                LevenshteinDistance(s, s_len - 1, t, t_len - 1) + cost)
