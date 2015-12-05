import csv

def result_writer(result,filename):
    """

    Writing Results

    :param result:
    :return:
    """
    with open('data/'+filename, 'w') as fp:
        writer = csv.writer(fp, delimiter=',')
        writer.writerow(['userId,testItems'])
        writer.writerows(result)
    fp.close

