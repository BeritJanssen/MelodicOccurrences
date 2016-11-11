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
import itertools as iter
import similarity as sim
import copy
import operator               

def annotated_phrase_identity(phrase1, phrase2, annotator_keys) :
    """ function called to check whether the label for phrase1 and phrase2 is 
    the same. Returns 1 if it is, 0 if it is not.
    """
    similarity_dict = {}
    for a in annotator_keys :
        if phrase1[a]==phrase2[a] :
            similarity_dict[a]=1
        else :
            similarity_dict[a]=0
    return similarity_dict

def filter_results(result_list, threshold, greater_or_lower, 
     sim_measure):
    """ takes a list of similarity results for finding occurrences, 
    a list of the results associated with annotated occurrences 
    and a threshold above (operator.gt) or below (operator.lt) which results 
    should be filtered"""
    filtered_list = [r for r in result_list if 
     greater_or_lower(r['matches'][sim_measure][0]['similarity'],threshold) 
     and r['query_filename']!=r['match_filename']]
    return filtered_list

def prepare_position_evaluation(result_list, mel_dict, label_dict, sign):
    """ for each result, find the annotated occurrences and tag them in the 
    match melody, together with the best algorithmic matches
    sign indicates whether the default should be extremely high or low """
    annotator_keys = ('ann1', 'ann2', 'ann3')
    output_keys = ('query_filename','query_segment_id',
     'match_filename','tunefamily_id')
    position_ev = []
    for r in result_list:
        matched_phrase = next((s for s in label_dict if 
         s['filename'] == r['query_filename'] 
         and int(s['phrase_id']) == r['query_segment_id']), None)
        match_melody = next((m for m in mel_dict if 
         m['filename']==r['match_filename']), None)
        algkeys = r['matches'].keys()
        # initiate the melody as containing 
        # only extremely high or low values
        evaluation_melody = [{'onset':m['onset'],'phrase_id':m['phrase_id']} 
         for m in match_melody['symbols']]
        phrase_id = 0
        # get the label of the first phrase in the melody
        relevant_phrase_label = next((s for s in label_dict if 
         s['filename']==r['match_filename'] and 
         int(s['phrase_id'])==0), None)
        comparison = annotated_phrase_identity(matched_phrase, 
         relevant_phrase_label, annotator_keys)
        for alg in algkeys:
            for m in r['matches'][alg]:
                similarity = m['similarity']
                start_onset = m['match_start_onset']
                end_onset = m['match_end_onset']
                for ev in evaluation_melody:
                    if start_onset<=ev['onset']<=end_onset:
                        ev[alg] = similarity
            for ev in evaluation_melody:
                if not alg in ev:
                    ev[alg] = sign * 16000
        for ev in evaluation_melody:
            if ev['phrase_id']!=phrase_id:
                relevant_phrase_label = next((s for s in label_dict if 
                 s['filename']==r['match_filename'] and 
                 int(s['phrase_id'])==ev['phrase_id']),None)
                phrase_id = ev['phrase_id']
                comparison = annotated_phrase_identity(matched_phrase, 
                 relevant_phrase_label, annotator_keys)
            for ann in annotator_keys:
                ev[ann] = comparison[ann]
            ann_sum = ev['ann1']+ev['ann2']+ev['ann3']
            ev['majority'] = 0
            ev['all'] = 0
            if ann_sum == 3:
                ev['majority'] = 1
                ev['all'] = 1
            elif ann_sum == 2:
                ev['majority'] = 1  
        this_ev = {key:r[key] for key in output_keys}
        this_ev['position_eval'] = evaluation_melody
        position_ev.append(this_ev)
    return position_ev

