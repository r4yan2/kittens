# kittens

**2017/18**

## Random

Really nothing to say, the algorithm try 5 random items which have not been evaluated by the user

## Top-N recommendations

* Basic recommendation algorithm, assign to every user the best-evaluated or the most-evaluated items
* The "advanced" one uses a variable shrink
* The "ultimate" one uses a variable shrink and is feature-aware


**2015/16**

## Top-N Personalized

__Features__

* variable shrink value 
$$|\log(\frac{\\#rating}{\\#users})| $$

* for every item in the topN, for every feature of that film, if the user has evaluated that feature the item rating get boosted of a value equal to $$ \frac {evaluation(user, feature)}{\frac { \\# times\ that\ user\ evaluated\ that\ specific\ feature}{\\#user\ evaluations}} $$

Than the result is sorted and the first 5 items are recommended

Is a successful approach is a TopN + CBF and has been proved to have a good base

Can be improved playing with the coefficient and the _point system_

## Cosine Similarity

applied to users...
failed!
should have been applied to items

## Pearson Similarity

Currently work in progress, it's known to be the best user-similarity

# Here is missing a huge part of our work, because of faggotry of the repo owner in first place, and monkey coding something that eventually was discovered not working well causing depression and shit storming in both coders who decided both to drop documentation

## Return to the CBF

Since the CF approach as gone wild, we now came back to a pure and simple CBF in which the binary based approach is applied to all the dataset.

Applied to all users the binary based:

'''
def get_binary_based_recommendations(user):
    recommendations = []
    for item in get_user_evaluation_list(user):
        features = get_features_list(item)
        num_features = len(features)
        if num_features == 0:
            continue
            tf = 1.0 / num_features
            tf_idf = map(lambda feature: get_features_global_frequency(feature) * tf, features)
            similarities = []
            for item_iterator in get_item_set():
                if item == item_iterator:
                    continue
                features_item_iterator = get_features_list(item_iterator)
                binary_features = map(lambda x: 1 if x in features_item_iterator else 0, features)
                num_features_item_iterator = len(features_item_iterator)
                if num_features_item_iterator == 0:
                    continue
                similarities.append([item_iterator,sum([a * b for a, b in zip(binary_features, tf_idf)]) / num_features_item_iterator])
            recommendations.append(sorted(similarities, key=lambda x: x[1], reverse=False))
 return recommendations
 '''python

298 Points

The result are taken as a round robin of the columns of recommendations

Changing the way recommendation are taken, so sorting by the higher similairty value the result obtained is: 307

Using the CF and than the CBF over the possible films to recommend, result obtained is: 319
Using the new_kittens_recommendations and binary for the users with more than 2 films seen, result obtained is: 327
