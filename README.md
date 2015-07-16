These are the implementations of similarity measures for finding occurrences of melodic segments in melodies.

music_representations.py comprises functions to convert from **kern to the representation used for the measures: a dictionary which contains the pitch histogram, file name, tune family id and for each note, its pitch, onset, duration, etc. This module relies on music21.

find_matches.py performs the comparison of melodic segments to melodies through distance measures, local alignment and SIAM. The function "matches_in_corpus" is used to order the corpus per tune family, and for one selected comparison method, finds the best matches of each query segment within each melody.

simarity.py collects different distance measures and the actual alignment algorithm, with different substitution functions.

Copyright 2015, Berit Janssen.