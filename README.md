# computerpraktikum-maschinelles-lernen

Python version: 3.7

## Modules:
* numpy 1.16.4 (for general (vectorized) computations)
* matplotlib 3.0.3 (for the plots)
* tkinterx 0.0.9 (for the GUI)
* scikit-learn 0.23.1 (for comparing to an existing implementation - not used for own computing)

## Getting started
* Execute `main.py`
* A GUI should open
* Select a data directory containing the datasets in the same format as the provided ones (by default `./data` is assumed)
* Now you can adjust various parameters (the maximum value of k to be tried, the partition count l, the search algorithm) for the classification algorithm
* Then either select a single dataset and run the classification algorithm for it (the algorithm has finished once a dialog opens announcing that it did complete) or run the classification for all datasets - then the results will be print on the console (otherwise you'll get multiple plots and some data in the GUI).
* Avalible nearest neighbor searches are brute sorting and k-d tree, however the latter is not sufficiently optimized.

