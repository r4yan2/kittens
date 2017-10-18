# kittens

The Engine is by default parallel but it's optimized to consume less RAM with respect to a normal parallel.

## Random

Really nothing to say, the algorithm try 5 random items which have not been evaluated by the user

## Top-N recommendations

Basic Top-N, assign to every user the most-listened or the most-included tracks

## Tags Based recommendations

A Tags based recommendations which takes into account the tags of a track with respect to the tags of a playlist.
The number of matching tags of the tracks are divided by the number of tags of the playlist.
Another index is computed as number of matching tags divided by the number of track tags.
The final recommendations are calculated sorting the tracks by the first index and in case of parity taking the second

## TF-IDF Recommendations

Implemented the tf-idf recommendation method.
Is performed by taking the tracks of the playlist as a whole, considering all tags and comparing them with the tracks to recommend

### TF-IDF (bad slow one)

This consider every track of the playlist and make an average at the end. Is slower and score less that the first so for the moment is discarded

## Tf-IDF pre-filtered with top tags

## Advanced Options

### Test-Utils

Is it possible to run the Engine in "Test Mode", which mean that:

* "some" row are randomly taken away from train set to form the test set
* consequently train is used without that test set
* after all the recommendations are computed, tests are run to measure the average precision of the chosen algorithm

### Script-utils

Is possible to run some script to pre-compute some data maps, store it into a csv file for later retrival

## Table of results

|Algorithm|Score|MAP@5|Precision|Recall|
|:-------:|:---:|:---:|:----:|:-------:|
|Random|0.0000|NA|NA|NA|
|top-listened|0.0002|NA|NA|NA|
|top-included|0.0100|NA|NA|NA|
|tags based recommendations|NA|NA|NA|NA|
|tf-idf recommendations|NA|NA|NA|NA|
|tf-idf (bad one)|NA|0.02546|0.02517|0.03644|
|top-tag ->(50)-> tf-idf|NA|NA|NA|NA|
|top-tag ->(75)-> tf-idf|NA|NA|NA|NA|
|top-tag ->(100)-> tf-idf|NA|NA|NA|NA|
|top-tag ->(125)-> tf-idf|NA|NA|NA|NA|
|tf-idf ->(50)-> top-tag|NA|NA|NA|NA|
|tf-idf ->(75)-> top-tag|NA|NA|NA|NA|
|tf-idf ->(100)-> top-tag|NA||NA|NA|NA|
|tf-idf ->(125)-> top-tag|NA|NA|NA|NA|
top-tag ->(75) -> tf_idf_titles|NA|NA|NA|NA|
top-tag ->(150) -> tf_idf_titles|NA|NA|NA|NA|
top-tag ->(350) -> tf_idf_titles|NA|NA|NA|NA|
top-tag ->(200) -> tf_idf_titles|NA|NA|NA|NA|
top-tag ->(250) -> tf_idf_titles|NA|NA|NA|NA|
