similarities = []

for itemI in itemSet:

    sys.stdout.write("\rMaking Item %i" % itemI)
    sys.stdout.flush()
    
    for itemJ in itemSet:
        if (itemJ > itemI):
            usersI = get_item_evaluators_list(itemI)[:]  # take the copy of the users that evaluated itemI
            usersJ = get_item_evaluators_list(itemJ)[:]  # take the copy of the users that evaluated itemJ

            ratingsItemI = []  # will contain the evaluations of User of the common items with userIterator
            ratingsItemJ = []  # will contain the evaluations of userIterator of the common items with userIterator

            for user in usersJ:
                if user in usersI:
                    ratingsItemI.append(get_evaluation(user,itemI) - avgUserRating[user])
                    ratingsItemJ.append(get_evaluation(user,itemJ) - avgUserRating[user])
            if not (len(ratingsItemI) == 0):
                shrink = math.fabs(math.log(float(len(ratingsItemI)) / get_num_users()))
                similarity = pearson_item_based_correlation(itemI, itemJ, ratingsItemI, ratingsItemJ, shrink)

                if similarity > 0.60:  # taking into consideration only positive and significant similarities
                    similarities.append([itemI,itemJ,similarity])

with open('data/item_similarities.csv', 'w') as fp:
    writer = csv.writer(fp, delimiter=',')
    writer.writerows(list(similarities))
fp.close
