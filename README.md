# kitt<3ns

The Engine is by default parallel. To have further info on the methods please refer to the specific method built-in documentation

## Random

Really nothing to say, the algorithm try 5 random items which have not been included by the user in his playlists

## Top-N recommendations

Basic Top-N, assign to every user the most-listened or the most-included tracks

## Tags Based recommendations

A Tags based recommendations which takes into account the tags of a track with respect to the tags of a playlist.
The number of matching tags of the tracks are divided by the number of tags of the playlist.
Another index is computed as number of matching tags divided by the number of track tags.
The final recommendations are calculated sorting the tracks by the first index and in case of parity taking the second

## TF-IDF Recommendations

Implemented the tf-idf recommendation method.
It is performed by taking the tracks of the playlist as a whole, considering all tags and comparing them with the tracks to recommend

### TF-IDF (bad slow one)

This consider every track of the playlist and make an average at the end. It is slower and score less than the first, so for the moment is discarded

## Neighborhood based

The idea is to restrict the selection of target tracks to the nearest neighborhood, selected by tdf-idfing the tags of the tracks of the playlists and selecting the playlist with the highest value of similarity

## Bayes Recommendations

The idea is to implement the bayes classifier to make the recommendations. The probability to include a track into the playlist is computed by taking into account how many tags of the target track matches that on the playlist, and then the tracks with best results are recommended.


## Advanced Options

### Test-Utils

Is it possible to run the Engine in "Test Mode", which mean that:

* "some" rows are randomly taken away from train set to form the test set
* consequently train is used without that test set
* after all the recommendations are computed, tests are run to measure the metrics of the choosen algorithm

### Script-utils

It is possible to run some script to pre-compute some data maps, store it into a csv file for later retrival

### Debug-mode

It is possible to run the engine in debug mode, using only a single core and sending output to terminal for easier debugging of newer methods

### results
KNN = 50, Cosine similarity: [['MAP@5', 0.08836894444444429], ['Precision', 0.07927999999999846], ['Recall', 0.11659921008729825]]
KNN = 50, Jacard similarity: [['MAP@5', 0.08927627777777772], ['Precision', 0.08107999999999835], ['Recall', 0.11767104167074746]]
KNN = 75, Jacard similarity: ['MAP@5', 0.09094077777777773], ['Precision', 0.08291999999999831], ['Recall', 0.12050940969161551]] -> scored about 0.06800
KNN = 75, Cosine similarity: ['MAP@5', 0.08981249999999992], ['Precision', 0.08103999999999838], ['Recall', 0.11927983612792434]]
KNN = 125, Jacard similarity:['MAP@5', 0.09120927777777771], ['Precision', 0.08351999999999828], ['Recall', 0.12145908388567055]]


\* means outdated
