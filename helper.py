import csv
import math
from operator import mul
import logging
try:
    import matplotlib.pyplot as plt
except ImportError:
    pass

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
    try:
        if len(content) > 1:
            writer.writerows(content)
        else:
            writer.writerow(content)
    except Exception as e:
        logging.debug("%s" % (e))
        fp.write(','.join([str(item) for item in content]))
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
    return csv.reader(open('data/'+what+'.csv', 'rb'), delimiter=delimiter)

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

def square_sum(x):
    """
    Squares each element of the list and returns the sum

    :param x: list of element to square
    :return: result list
    """
    acc = 0
    for elem in x:
        acc += elem*elem
    return acc

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

def set_difference_len(I,J):
    """
    Compute the difference between two set
    :param I: list/set
    :param J: list/set
    :return: number of different elements
    """
    if isinstance(I,set):
        return len(I.symmetric_difference(J))
    elif isinstance(J, set):
        return len(J.symmetric_difference(I))
    else:
        acc=0
        for i in I+J:
            if not (i in J and i in I):
                acc+=1
        return acc

def jaccard(I, J):
    """
    Compute the jaccard similarity improved y multiplication with the mse

    :param I: set
    :param J: set
    :return: similarity value
    """
    if not isinstance(I, set):
        I = set(I)
    if not isinstance(J, set):
        J = set(J)
    intersection = float(len(I.intersection(J)))
    if not intersection:
        return 0.0
    union = float(len(I.union(J)))
    if not union:
        return 0.0
    disjoint = float(len(I.symmetric_difference(J)))
    if disjoint == union:
        return 0.0
    #ratseq = [math.fabs(tracks.index(i) - tracks.index(j)) for playlist in I.intersection(J) for tracks in [db.get_playlist_tracks(playlist)]]
    jaccard_coefficient = intersection / union
    mse = 1.0 - disjoint / union

    return jaccard_coefficient * mse

def parseIntList(lst):
    """
    Manual parsing for some fields of the csv, help to avoid uses of eval function

    :param lst: string to parse
    :return: parsed integer list
    """
    return [int(num) for num in lst.strip('[]\n').split(',') if num != 'None' and num != '']

def parseFloatList(lst):
    """
    Manual parsing for some fields of the csv, help to avoid uses of eval function

    :param lst: string to parse
    :return: parsed integer list
    """
    return [float(num) for num in lst[1:-1].split(',') if num != 'None' and num != '']

def phi_coefficient(list1, list2, tot_tags):
    """
    Compute the Phi similarity coefficient between the lists taken in input
    :param list1
    :param list2
    """
    # a = proportion of 1s that the variables share in the same positions
    # b = proportion of 1s in the first variable and 0s in second variable in the same positions
    # c =  proportion of 0s in the first variable and 1s in second variable in the same positions
    # d = proportion of 0s that both variables share in the same positions

    a = 0
    b = 0
    c = 0
    d = 0

    for i in list1 + list2:
        if i in list1 and i in list2:
            a += 1
        elif i in list1 and i not in list2:
            b += 1
        elif i not in list1 and i in list2:
            c += 1
    d = tot_tags - (a + b + c)
    try:
        phi = (a * d - b * c) / math.sqrt((a + b) * (a + c) * (b + d) * (c + d))
    except ZeroDivisionError:
        phi = 0
    return phi


def hemming_distance(a, b):
    """
    
    """
    counter = 0
    equal = 0
    for elem in a:
        if elem in b: 
            equal += 1
        else: 
            counter += 1
    return counter + len(b) - equal 

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

def plot(lst):
    """
    Helper function to easy plot data stored from kittens past execution
    
    :param lst: name or list of names of saved map5 distributions
    """
    if not isinstance(lst, list):
        if not isinstance(lst, str):
            raise ValueError("Input is not a string")
        to_draw = [[int(length), float(value)] for value, length in read(lst, ',')]
        plt.plot(*zip(*to_draw), label=lst)
    else:
        for elem in lst:
            if not isinstance(elem, str):
                raise ValueError("Input is not a string")
            to_draw_value = (float(value) for value, length in read(elem, ','))
            to_draw_length = [int(length) for value, length in read(elem, ',')]
            to_draw_pos = set(to_draw_length)
            to_draw = [[i, next(to_draw_value) if i in to_draw_pos else 0.0] for i in xrange(1,len(to_draw_length))]
            #to_draw = [[int(length), float(value)] for value, length in read(elem, ',')]
            plt.plot(*zip(*to_draw), label=elem)
    plt.legend(loc='best')
    plt.show(block=False)
