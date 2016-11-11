"""
    Copyright 2015, Berit Janssen.

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
from scipy import spatial

def cardinality_score(seq1, seq2):
    """ calculates the cardinality score between two sequences """
    rset = set([(r['onset'],r['pitch']) for r in seq1])
    qset = set([(q['onset'],q['pitch']) for q in seq2])
    cSc = len(rset.intersection(qset))
    return cSc

def city_block_distance(seq1, seq2):
    """ calculates the city-block distance between two sequences"""
    dif = spatial.distance.cityblock(seq1, seq2)
    return dif/float(len(seq1))

def correlation(seq1, seq2):
    """ calculates the correlation distance between two sequences """
    # returns 1 - correlation coefficient
    cor = spatial.distance.correlation(seq1, seq2)
    return cor

def euclidean_distance(seq1, seq2):
    """ calculates the euclidean distance between two sequences """
    sim = spatial.distance.euclidean(seq1, seq2)
    return sim/float(len(seq1))

def hamming_distance(seq1, seq2): 
    """ calculates the hamming distance between two sequences """
    mm = spatial.distance.hamming(seq1, seq2)
    return mm

def match_distance_tuples(seq1, seq2, dist_func): 
    """returns the distance for multidimensional data points, using the chosen 
    distance function (dist_func) in scipy.spatial"""
    mm = spatial.distance.cdist(np.array(seq1), np.array(seq2), dist_func)
    return mm

def multi_dimensional(seq1, seq2, variances): 
    """euclidean distance of the points in local alignment"""
    return -spatial.distance.seuclidean(seq1, seq2, variances) + 1.0

#seq1, seq2: array of symbols (dictionaries)
#gap_score: float
#sim_score: function that takes two symbols and returns float
def local_alignment(seq1, seq2, insert_score, delete_score, sim_score, 
    return_positions, variances=[]):
    """ local alignment takes two sequences (the query comes first), 
    the insertion and deletion sore,
    and a function which defines match / mismatch
    returns the index where the match starts in the sequence, 
    and the normalized score of the match
    """
    #initialize dynamic programming matrix
    d = np.zeros([len(seq1)+1, len(seq2)+1])
    #initialize backtrace matrix
    b = np.zeros([len(seq1)+1, len(seq2)+1])
    max_score = 0.0
    #fill dynamic programming matrix
    for i in range(1, len(seq1) + 1):
        # query sequence, rows of dynamic programming matrix
        for j in range(1, len(seq2) + 1):
            # matched in longer sequence, columns of dynamic programming matrix
            from_left = d[i,j-1] + delete_score
            from_top = d[i-1,j] + insert_score
            diag = d[i-1,j-1] + sim_score(seq1[i-1], seq2[j-1], variances)
            d[i,j] = max(from_top, from_left, diag, 0.0)
            if d[i,j] > max_score:
                max_score = d[i,j]
            # store where the current entry came from in the backtrace matrix
            if d[i,j] == from_left:
                # deletion from longer sequence
                backtrace = 0
            elif d[i,j] == from_top:
                # insertion into longer sequence
                backtrace = 1
            elif d[i,j] == diag:
                # substitution
                backtrace = 2
            else:
                backtrace = -1
            b[i,j] = backtrace
    m,n = np.where(d == max_score)
    # convert from numpy array to integer
    similarity = max_score/float(len(seq1))
    if not return_positions:
        return [int(n[0]), 0, similarity]
    # store the length of the match as well to return 
    #(match length can be shorter than query length)
    match_list = []
    # do not return more than 5 matches
    if m.size>4:
        num_matches = 5
    else:
        num_matches = m.size
    for i in range(num_matches):
        row = int(m[i])
        column = int(n[i])
        match_length = 0
        while d[row,column] > 0 :
            if b[row,column] == 0:
                # deletion from longer sequence, move left
                column -= 1
                match_length += 1
            elif b[row,column] == 1:
                # insertion into longer sequence, move up
                row -= 1
            elif b[row,column] == 2:
                # substitution, move diagonally
                row -= 1
                column -= 1
                match_length += 1
            else :
                print(d, b, row, column)
        match_list.append([column, match_length, similarity])
    return match_list

def pitch_rater(seq1, seq2, variances):
    """ subsitution score for local alignment"""
    if seq1 == seq2:
        return 1.0
    else:
        return -1.0

def pitch_difference(seq1, seq2, variances):
    """ subsitution score for local alignment: 
    returns the difference between pitches in two sequences"""
    return 2.0 - abs(seq1-seq2)

def label_diff(seq1, seq2) :
    """ called by ir_alignment """
    if seq1['IR_structure'] == seq2['IR_structure']: 
        return 0.0
    elif seq1['IR_structure'].strip('[]') == seq2['IR_structure'].strip('[]'): 
        return 0.801
    else: 
        return 1.0

def ir_alignment(seq1, seq2, variances): 
    """ substitution score for IR structure alignment """
    subsScore = .587*label_diff(seq1, seq2) + .095*abs((seq1['end_index'] - 
        seq1['start_index']) - (seq2['end_index'] - seq2['start_index']))
    + .343*abs(seq1['direction'] - seq2['direction'])
    + .112*abs(seq1['overlap'] - seq2['overlap'])
    return 1.0 - subsScore
 