# kittens

The Engine is by default parallel. To have further info on the methods please refer to the specific method built-in documentation

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

## Neighborhood based

The ide is to restrict the selection of target tracks to the nearest neighborhood, selected by tdf-idfing the tags of the tracks of the playlists and selecting the playlist with the highest value of similarity

## Advanced Options

### Test-Utils

Is it possible to run the Engine in "Test Mode", which mean that:

* "some" row are randomly taken away from train set to form the test set
* consequently train is used without that test set
* after all the recommendations are computed, tests are run to measure the metrics chosen algorithm

### Script-utils

Is possible to run some script to pre-compute some data maps, store it into a csv file for later retrival

### Debug-mode

Is possible to run the engine in debug mode, using only a single core and sending output to terminal for easier debugging of newer methods

## Table of results

|Algorithm|Score|MAP@5|Precision|Recall|test-machine|
|:-------:|:---:|:---:|:----:|:-------:|:----------:|
|top-listened|0.0002|NA|NA|NA|traveller|
|top-included|0.0100|0.000954|0.00152|0.0016|traveller|
|tags based recommendations|NA|0.01569|0.01352|0.02512|traveller|
|tf-idf recommendations|NA|0.03894|0.03515|0.05536|traveller|
|tf-idf recomm with album and artits (0<x<60000)|NA|0.075692|0.06995|0.1000|traveller|
|tf-idf recomm with album and artits (0<x<30000)|NA|0.077111|0.06999|0.09998|traveller|
|tf-idf recomm with album and artits (0<x<15000)|NA|0.073720|0.06811|0.09695|traveller|
|tf-idf (bad one)|NA|0.02546|0.02517|0.03644|traveller|
|top-tag ->(50)-> tf-idf|NA|NA|NA|NA|
|top-tag ->(75)-> tf-idf|NA|NA|NA|NA|
|top-tag ->(100)-> tf-idf|NA|NA|NA|NA|
|top-tag ->(125)-> tf-idf|NA|NA|NA|NA|
|tf-idf ->(50)-> top-tag|NA|NA|NA|NA|
|tf-idf ->(75)-> top-tag|NA|NA|NA|NA|
|tf-idf ->(100)-> top-tag|NA|NA|NA|NA|
|tf-idf ->(125)-> top-tag|NA|NA|NA|NA|
|top-tag ->(75)-> tf_idf_titles|NA|NA|NA|NA|
|top-tag ->(150)-> tf_idf_titles|NA|NA|NA|NA|
|top-tag ->(350)-> tf_idf_titles|NA|NA|NA|NA|
|top-tag ->(200)-> tf_idf_titles|NA|NA|NA|NA|
|top-tag ->(250)-> tf_idf_titles|NA|NA|NA|NA|
|tf-idf artist w/ fallback tf-idf tags w/ fallback tf-idf listened|NA|0.032192|0.03104|0.046985|traveller|

\* means outdated
