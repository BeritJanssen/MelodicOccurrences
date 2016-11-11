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
    in the specified music representation.
    A scaling factor can be used to determine the correct positions 
    of a duration weighed pitch curve 
    """
    result_list = []
    keys = ['cbd','ed','cd']
    for seg in segment_list:
        segment_curve = [a[music_representation] for a in seg['symbols']]
        if segment_curve[0] is None:
            # checking if first note has an undefined value, 
            # e.g. pitch interval
            segment_curve = segment_curve[1:]
        query_length = len(segment_curve)
        for mel in melody_list: 
            result_dict = {}
            matches = []
            mel_curve = [a[music_representation] for a in mel['symbols']]
            if mel_curve[0] is None:
                # if the first value (e.g. pitch interval, ioi) 
                #is undefined, discard it
                mel_curve = mel_curve[1:]
            if query_length > len(mel_curve):
                # for duration weighed sequences,
                # query sequences might be longer than a melody sequence
                this_segment_curve = segment_curve[0:len(mel_curve)]
                len_to_slide = len(mel_curve) - len(this_segment_curve) + 1
            else :
                this_segment_curve = segment_curve
                len_to_slide = len(mel_curve) - query_length + 1
            for l in range(len_to_slide): 
                mel_segment = mel_curve[l:query_length+l]
                if not np.std(this_segment_curve) or not np.std(mel_segment):
                    cor = np.nan
                else:
                    cor = sim.correlation(this_segment_curve,mel_segment)
                city = sim.city_block_distance(this_segment_curve,mel_segment)
                euclid = sim.euclidean_distance(this_segment_curve,mel_segment)
                matches.append({'cbd':city,'ed':euclid,'cd':cor})
            for k in keys: 
                best_similarity = np.nanmin([m[k] for m in matches])
                best_match_indices= [i for i,m in enumerate(matches) 
                 if m[k]==best_similarity]
                match_list = []
                for b in best_match_indices:
                    match_stats = {'similarity': best_similarity} 
                    if return_positions:
                        match_start_onset, match_end_onset = find_positions( 
                         mel, b, query_length-1, scaling)
                        match_stats['match_start_onset'] = match_start_onset
                        match_stats['match_end_onset'] = match_end_onset
                    match_list.append(match_stats)
                result_dict[k] = match_list 
            result_list.append({'tunefamily_id':seg['tunefamily_id'],
            'query_filename':seg['filename'],
            'match_filename':mel['filename'],
            'query_segment_id':seg['segment_id'],
            'query_length':query_length,
            'matches':result_dict})
    return result_list

def find_positions(melody_dict, match_index, query_length, scaling):
    if not scaling:
        match_start_onset = melody_dict['symbols'][match_index]['onset']
        match_end_onset = (melody_dict['symbols']
         [match_index + query_length]['onset'])
    else:
        match_start_onset = match_index / float(scaling)
        match_end_onset = (match_index + query_length) / float(scaling)
        if 'onsets_multiplied_by' in melody_dict:
            match_start_onset = (match_start_onset / 
             float(melody_dict['onsets_multiplied_by']))
            match_end_onset = (match_end_onset / 
             float(melody_dict['onsets_multiplied_by']))
    return match_start_onset, match_end_onset

def local_aligner(melody_list, segment_list,
 music_representation, return_positions, scaling, insertion_weight=-.5,
 deletion_weight=-.5, substitution_function=sim.pitch_rater, variances=[]):
    """ this function takes melodies and segments belonging to the same 
    tune family, represented as lists of dictionaries,
    and finds occurrences using local alignment, 
    in the specified music representation.
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
        segment_curve = [a[music_representation] for a in seg['symbols']]
        query_length = len(segment_curve)
        if segment_curve[0] is None: 
		    # if the first value (e.g. pitch interval, ioi) 
            #is undefined, discard it
            segment_curve = segment_curve[1:]
        for mel in melody_list: 
            mel_curve = [a[music_representation] for a in mel['symbols']]
            if mel_curve[0] is None:
			    # if the first value (e.g. pitch interval, ioi) 
                #is undefined, discard it
                mel_curve = mel_curve[1:]
            match_list = sim.local_alignment(
             segment_curve, mel_curve,
             insertion_weight, deletion_weight,
             substitution_function, return_positions, variances)
            match_results = [{'similarity': match_list[0][2]}]
            if return_positions:
                match = {'similarity': match_list[0][2]}
                match_results = []
                for m in match_list:
                    match_start_onset, match_end_onset = find_positions(mel, 
                     m[0], m[1] - 1, scaling)
                    match['match_start_onset'] = match_start_onset
                    match['match_end_onset'] = match_end_onset
                    match_results.append(match.copy())
            result_list.append({'tunefamily_id': mel['tunefamily_id'],
             'query_filename': seg['filename'],
             'match_filename': mel['filename'],
             'query_segment_id': seg['segment_id'],
             'query_length': query_length,
             'matches': {'la': match_results}})
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
        seg_points = [(s['onset'] - start_onset, s['pitch']) for 
         s in seg['symbols']]
        for mel in melody_list: 
            translation_vectors = []
            translation_vectors_with_position = []
            mel_points = np.array([(s['onset'], s['pitch']) for 
             s in mel['symbols']])
            for p in seg_points: 
                vectors = (mel_points - p)
                translation_vectors.extend([tuple(v) for v in vectors])
                translation_vectors_with_position.append((p[0], 
                 [tuple(v) for v in vectors]))
            grouped_vectors = dict(Counter(translation_vectors))
            # the similarity is the size of the maximal TEC
            similarity = max([grouped_vectors[k] for k in grouped_vectors])
            match_results = {'similarity': similarity / float(len(seg_points))} 
            if return_positions:
                match = {'similarity': similarity / float(len(seg_points))}
                match_results = []
                shifts = [key for key in grouped_vectors if 
                 grouped_vectors[key]==similarity]
                for shift in shifts:
                    onsets = [vec[0] for vec in 
                     translation_vectors_with_position if shift in vec[1]]
                    match_start_onset = min(onsets) + shift[0]
                    match_end_onset = max(onsets) + shift[0]
                    if 'onsets_multiplied_by' in mel:
                        match_start_onset = (match_start_onset / 
                         float(mel['onsets_multiplied_by']))
                        match_end_onset = (match_end_onset / 
                         float(mel['onsets_multiplied_by']))
                    match['match_start_onset'] = match_start_onset
                    match['match_end_onset'] = match_end_onset
                    match_results.append(match.copy())
            result_list.append({'tunefamily_id': seg['tunefamily_id'],
             'query_filename': seg['filename'],
             'match_filename': mel['filename'],
             'query_segment_id': seg['segment_id'],
             'query_length': len(seg_points),
             'matches': {'siam': match_results}})
    return result_list
		
def matches_in_corpus(all_melody_list, all_segment_list,
 music_representation='pitch', measure=local_aligner, return_positions=True, 
 scaling=None, *args):
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