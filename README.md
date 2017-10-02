# kittens

The Engine is by default parallel but it's optimized to consume less RAM with respect to a normal (stupid) parallel.

## Random

Really nothing to say, the algorithm try 5 random items which have not been evaluated by the user (0.0000)

## Top-N recommendations

* Basic Top-N, assign to every user the most-listened (0.00012) or the most-included tracks

## Feature Based

TODO



## Advanced Options

### Test-Utils

Is it possible to run the Engine in "Test Mode", which mean that:

* "some" row are randomly taken away from train set to form the test set
* consequently train is used without that test set
* after all the recommendations are computed, tests are run to measure the average precision of the chosen algorithm

## Table of results

|Algorithm|Score|
|---|---|
|Random|0,0000|
|top-listened|0,0002|
