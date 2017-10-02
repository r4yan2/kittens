# kittens

## Random

Really nothing to say, the algorithm try 5 random items which have not been evaluated by the user

## Top-N recommendations

* Basic Top-N, assign to every user the most-listened or the most-included tracks

__Notes on top-n__

The algorithm need a certain threshold to avoid that film with low number of rating will bump in high places.
So a good threshold need to ensure that an item have at least a certain number of votes.
A threshold has been defined as x*max_votes, where x is the percentage representing the threshold and max_votes
is the maximum number of votes taken by a single film.
The problem with the threshold is that by using a too low value there is a few (but still significant) improvement.
If the threshol is high (80%+) the recommendation accuracy is HIGH, but there are no more than few film in the top-n
so a good compromise could be using top-n with high threshold in conjunction with other recommendations algorithm

## Advanced Options

### Test-Utils

Is it possible to run the Engine in "Test Mode", which mean that:

* 20980 row are randomly taken away from train set to form the test set
* consequently train is used without that test set
* after all the recommendations are computed, tests are run to measure the average precision of the chosen algorithm

### Debug Mode

Is it possible to specify a debug run for the application in which more stats are displayed during the run

### Parallel Processing

Is it possible to take advantage of a multi-core architecture splitting the user set by the number of core assigning
a portion of each core
_Note that the recommendations algorithm are parallelizable by design_
