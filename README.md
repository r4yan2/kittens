# kittens

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

## Current idea

graduatory in which we apply the base point 


if #seen by user > 5

$$ \frac {evaluation(user, feature)}{\frac { \\# times\ that\ user\ evaluated\ that\ specific\ feature}{\\#user\ evaluations}} $$

else recommendTopN to faggots

next we'll take the first K film and we will apply the item-based approach 


Current Evaluation

NeverSeen: rating = sum(features_ratings)/len(features_ratings)  
