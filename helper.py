import csv
import math
from operator import mul


def write(filename, content, delimiter_char=','):
    """
    write the result on the file

    :param filename: The name of the file to create
    :param content: The content to save
    :param delimiter_char: The separator char for fields (csv)
    :return: None
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

    :param a: One list
    :param b: Another list
    :return: a - b
    """
    b = set(b)
    return [aa for aa in a if aa not in b]

def diff_test_set(a, b):
    """
    make the difference between list of lists of type [[][]]

    :param a: One list of lists
    :param b: Another list of lists
    :return: a - b
    """
    b = set([str(x[0])+str(x[1]) for x in b])
    return [x for x in a if str(x[0])+str(x[1]) not in b]

def read(what, delimiter='\t'):
    """
    what should be an existing csv, return the csv reader to be listed

    :param what: The file to read
    :return: the reader object to parse
    """
    return csv.reader(open('data/'+what+'.csv'), delimiter=delimiter)

def cumulative_sum(lis):
    """
    Do a cumulative sum operation on a given list
    [1,2,3] -> [1,3,6]

    :param lis: the list to process
    :return: the cumulative sum list
    """

    total = 0
    for x in lis:
        total += x
        yield total

def multiply_lists(lst1, lst2):
    """
    Element wise multiplication

    :param lst1: First operand
    :param lst2: Second operand
    :return: list of multiplied elements
    """

    return [a * b for a, b in zip(lst1, lst2)]

def product(lst):
    return reduce(mul, lst, 1)

def divide_lists(lst1, lst2):
    """
    Element wise division
    Warning: if list of integer are used a list of integer is computed

    :param lst1: First operand
    :param lst2: Second operand
    :return: list of divided elements
    """
    return [a / b for a, b in zip(lst1, lst2)]

def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)

def square_sum (x):
    """
    Squares each element of the list and returns the sum

    :param x: list of element to square
    :return: result list
    """
    return sum([elem*elem for elem in x])


def pearsonr(x, y):
    """
    return the pearson correlation between x and y

    :param x: first vector
    :param y: second vector
    :return: the similarity coefficient
    :raise ValueError: if the two vectors have different length
    """
    length_x = len(x)
    length_y = len(y)
    if length_x != length_y:
        raise ValueError("The two arrays have different length")
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    normalized_x = [(elem - mean_x) for elem in x]
    normalized_y = [(elem - mean_y) for elem in y]

    numerator = sum(multiply_lists(normalized_x, normalized_y))
    denominator = math.sqrt(square_sum(normalized_x) * square_sum(normalized_y))

    return  numerator / denominator

def jaccard(I, J):
    """
    Compute the jaccard similarity improved y multiplication with the mse

    :param I: set
    :param J: set
    :return: similarity value
    """
    intersection = float(len(I.intersection(J)))
    union = float(len(I.union(J)))
    disjoint = float(len(I.union(J).difference(I.intersection(J))))
    jaccard_coefficient = intersection / union
    mse = 1.0 - disjoint / union
    return jaccard_coefficient * mse

def parseIntList(lst):
    """
    Manual parsing for some fields of the csv, help to avoid uses of eval function

    :param lst: string to parse
    :return: parsed integer list
    """
    return [int(num) for num in lst[1:-1].split(',') if num != 'None' and num != '']

def LevenshteinDistance(s, s_len, t, t_len):
    """
    Compute the LevenshteinDistance in recursive way(slow)

    :param s: list s
    :param s_len: length of list s
    :param t: list t
    :param t_len: length of list t
    :return: computed distance
    """
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
