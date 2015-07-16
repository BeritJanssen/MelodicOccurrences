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

import similarity as sim
import numpy as np
import time
from collections import Counter	

def distance_measures(melody_list,segment_list,
  music_representation,return_positions,scaling):
    """ this function takes melodies and segments belonging 
    to the same tune family, 
    represented as lists of dictionaries,
    and finds occurrences using a number of distance measures,
    in the specified music representation (specified by musicRep)
    A scaling factor can be used to determine the correct positions 
    of a duration weighed pitch curve 
    """
    result_list = []
    keys = ['hamming','city','euclid','cor']
    for seg in segment_list: 
        result_dict = {}
        segment_curve = [a[music_representation] for a in seg['symbols']]
        if not segment_curve[0] :
            # checking if first note has an undefined value, 
            # e.g. pitch interval
            segmentCurve = segment_curve[1:]
        for mel in melody_list: 
            matches = []
            mel_curve = [a[music_representation] for a in mel['symbols']]
            len_to_slide = len(mel_curve)-len(segment_curve)+1
            for l in range(len_to_slide): 
                if l==0: 
                    # checking if first note has an undefined value, 
                    # e.g. pitch interval
                    if not mel_curve[l]:
                        matches.append({'hamming':float('Inf'),
                         'city':float('Inf'),'euclid':float('Inf'),
                         'cor':float('Inf')})
                        continue
                mel_segment = mel_curve[l:len(segment_curve)+l]
                hamming = sim.hamming_distance(segment_curve,mel_segment)
                city = sim.city_block_distance(segment_curve,mel_segment)
                euclid = sim.euclidean_distance(segment_curve,mel_segment)
                cor = sim.correlation(segment_curve,mel_segment)
                matches.append({'hamming':hamming,
                 'city':city,'euclid':euclid,'cor':cor})
            for k in keys: 
                best_similarity = np.nanmin([m[k] for m in matches])
                result_dict[k] = {'similarity': best_similarity}
                if return_positions:
                    best_match_index = int(next((i for i,m in enumerate(matches) 
                     if m[k]==best_similarity),None)/scaling)
                    result_dict[k]['match_start_onset'] = (mel['symbols']
                     [best_match_index]['onset'])
                    result_dict[k]['match_end_onset'] = (mel['symbols']
                     [best_match_index + len(segment_curve)-1]['onset'])
            result_list.append({'tunefamily_id':seg['tunefamily_id'],
            'query_filename':seg['filename'],
            'match_filename':mel['filename'],
            'query_segment_id':seg['segment_id'],
            'query_length':int(len(segment_curve)/float(scaling)),
            'matches':result_dict})
    return result_list


def local_aligner(melody_list, segment_list,
 music_representations, return_positions, scaling=1.0, insertion_weight=-.5,
 deletion_weight=-.5, substitution_function=sim.pitch_rater, variances=[]):
    """ this function takes melodies and segments belonging to the same 
    tune family, represented as lists of dictionaries,
    and finds occurrences using local alignment.
    Different from SIAM and distance_measures, it accepts a list of music 
    representations, e.g. ['pitch'].
    The insertion and deletion weights define the gap penalties,
    the substitution function (in similarity.py) 
    defines the substitution rules.
    If the positions of the occurrences are requested, the scaling factor can be
    used to determine the correct positions for duration weighed pitch curves.
    Variances of note values (e.g. pitch variance) need to be given only if 
    several music representations are passed to the 
    local alignment for use in sim.multidimensional.
	"""
    result_list = []
    for seg in segment_list: 
        seg_curve = [[s[k] for k in music_representations] for 
         s in seg['symbols']]
        if not seg_curve[0][0]: 
		    # if the first value (e.g. pitch interval, ioi) 
            #is undefined, discard it
            seg_curve = seg_curve[1:]
        for mel in melody_list: 
            matches = []
            mel_curve = [[s[k] for k in music_representations] for 
             s in mel['symbols']]
            if not mel_curve[0][0]:
			    # if the first value (e.g. pitch interval, ioi) 
                #is undefined, discard it
                mel_curve = mel_curve[1:]
            match_index, match_length, similarity = sim.local_alignment(
             seg_curve, mel_curve,
             insertion_weight, deletion_weight,
             substitution_function, return_positions, variances)
            match_results = {'similarity':similarity}
            if return_positions:
                match_start_onset = (mel['symbols']
                 [int(round(match_index/scaling,0))]['onset'])
                match_end_onset = (mel['symbols']
                 [int(round(match_index/scaling,0))+match_length-1]['onset'])
                match_results['match_start_onset'] = match_start_onset
                match_results['match_end_onset'] = match_end_onset
            result_list.append({'tunefamily_id':mel['tunefamily_id'],
             'query_filename':seg['filename'],
             'match_filename':mel['filename'],
             'query_segment_id':seg['segment_id'],
             'query_length':len(seg_curve),
             'matches':{'local_align':match_results}})
    return result_list
	
def SIAM(melody_list,segment_list,music_representation,
 return_positions,scaling): 
    """ this function takes melodies and segments belonging 
    to the same tune family, 
	represented as lists of dictionaries,
	and finds occurrences using SIAM
	in the specified music representation (specified by music_representation)
    Optionally, the positions of the occurrences can be returned, if applicable,
    scaled by a scaling factor.
	"""
    result_list = []
    for seg in segment_list :
        start_onset = seg['symbols'][0]['onset']
        seg_points = [(s['onset']-start_onset,s['pitch']) for 
         s in seg['symbols']]
        for mel in melody_list: 
            translation_vectors = []
            translation_vectors_with_origin = []
            mel_points = np.array([(s['onset'],s['pitch']) for 
             s in mel['symbols']])
            for p in seg_points: 
                vectors = (mel_points - p)
                translation_vectors.extend([tuple(v) for v in vectors])
                translation_vectors_with_origin.append((p[0],[tuple(v) for 
                 v in vectors]))
            grouped_vectors = dict(Counter(translation_vectors))
            # the similarity is the size of the maximal TEC
            similarity = max([grouped_vectors[k] for k in grouped_vectors])
            match_results = {'similarity':similarity/float(len(seg_points))}
            if return_positions:
                shift = max(grouped_vectors, key=grouped_vectors.get)
                first_query_onset = next((vec[0] for vec in 
                 translation_vectors_with_origin if shift in vec[1]),None)
                match_start_onset = first_query_onset + shift[0]
                match_index = next((i for i,s in enumerate(mel['symbols']) if 
                 s['onset']>=match_start_onset),None)
                match_end_index = match_index+similarity-1
                if match_end_index>len(mel['symbols'])-1:
                    continue
                match_end_onset = mel['symbols'][match_end_index]['onset']
                match_results['match_start_onset'] = match_start_onset
                match_results['match_end_onset'] = match_end_onset
            result_list.append({'tunefamily_id':seg['tunefamily_id'],
             'query_filename':seg['filename'],
             'match_filename':mel['filename'],
             'query_segment_id':seg['segment_id'],
             'query_length':len(seg_points),
             'matches':{'siam':match_results}})
    return result_list
		
def matches_in_corpus(all_melody_list, all_segment_list,
 music_representation, measure=local_aligner, return_positions=True, 
 scaling=1.0, *args):
    """ this function finds occurrences in a corpus. It takes a list of 
    all melodies and segments in a corpus, and finds occurrences in the 
    specified music representation with the specified similarity measure.
    If return_positions is true, the position of the occurrences will be 
    returned as well, at the expense of computation time.
    For duration weighted pitch sequences, the scaling factor indicates 
    the sampling of the pitch sequences, and makes it possible to recalculate 
    the position in quarterLength (cf. music21). """
    tick = time.clock()
    all_results = []
    tune_fams = set([m['tunefamily_id'] for m in all_melody_list])
    for fam in tune_fams :
        segment_list = [s for s in all_segment_list if s['tunefamily_id']==fam]
        melody_list = [m for m in all_melody_list if m['tunefamily_id']==fam]
        print(fam, len(melody_list), len(segment_list))
        fam_results = measure(melody_list, segment_list, music_representation, 
         return_positions, scaling, *args)
        all_results.extend(fam_results)
    print(time.clock()-tick)
    return all_results