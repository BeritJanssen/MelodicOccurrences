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
from sklearn import metrics
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
            continue
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
     greater_or_lower(r['matches'][sim_measure]['similarity'],threshold)]
    return filtered_list

def order_errors_by_tunefamily(result_list, phrase_dict):
    """ takes a result list of errors (false positives / false negatives), 
    and the dictionary of matched phrases for normalization purposes """
    ordered_errors_dict = []
    tunefams = set([p['tunefamily_id'] for p in phrase_dict])
    for t in tunefams:
        phrases = [p for p in phrase_dict if p['tunefamily_id'] == t]
        melodies = set([p['filename'] for p in phrases])
        num_errors = len([r for r in resultList if r['id'][:12] in melodies])
        normalize = len(phrases)
        ordered_errors_dict.append({'tunefamily_id': t, 
         'percentage': num_errors / float(normalize)})
    return ordered_errors_dict

def pattern_precision_recall(filtered_results, mel_dict, sim_dict, 
     sim_measure, annotator):
    """ return precision and recall,
    as defined by David Meredith for Three-Layer measures (first level) 
    takes the full result list with positions of matches,
    a selection of results (e.g. true positives), a dictionary of the melodies 
    with original onset positions etc., 
    and for the given similarity measure and annotator
    calculates the agreement of pattern position"""
    pattern_precision_recall = []
    for filt in filtered_results:
        ann_occurrences = []
        alg_start = filt['matches'][sim_measure]['match_start_onset']
        alg_end = filt['matches'][sim_measure]['match_end_onset']
        alg_length = alg_end - alg_start
        # get the melody in which the query is matched
        mel = next((m['symbols'] for m in mel_dict 
         if m['filename'] == filt['match_filename']), None)
        alg_match = [s for s in mel if alg_start<= s['onset'] <= alg_end]
        matched_phrase = next((s for s in sim_dict 
         if s['filename'] == filt['query_filename'] and 
         int(s['phr_id']) == filt['query_segment_id']), None)
        ann_labels_this_melody = [s for s in sim_dict if 
         s['filename']==filt['match_filename']]
        for l in ann_labels_this_melody:
            occurrence = annotatedPhraseSimilarityBinary(matched_phrase,
             l, [annotator])[annotator]
            if occurrence==1:
                ann_occurrences.append(int(l['phr_id']))
        if ann_occurrences:
            max_overlap = 0
            for ann in ann_occurrences:
                comparisons = []
                ann_match = [s for s in mel if s['phrase_id'] == ann]
                note_overlap = sim.cardinality_score(alg_match,ann_match)
                if note_overlap > max_overlap:
                    max_overlap = note_overlap
                    ann_length = len(ann_match)
        else:
            continue
        alg_length = len(alg_match)
        recall = max_overlap / float(ann_length)
        precision = max_overlap / float(alg_length) 
        pattern_precision_recall.append({'precision': precision, 
         'recall': recall, 'query_filename': filt['query_filename'], 
         'query_segment_id': filt['query_segment_id'], 
         'match_filename': filt['match_filename']})
    return pattern_precision_recall

def prepare_evaluation(result_dict, label_dict) :
    """ takes the result of segment matching within melodies
    and a dictionary of annotated phrase labels
    returns a list of dictionaries with the annotated occurrence (1 or 0) 
    and the calculated similarity
    """
    annotator_keys = ('ann1', 'ann2', 'ann3')
    matches_and_annotations = []
    tunefams = set([a['tunefamily_id'] for a in label_dict])
    for r in result_dict :
        matched_phrase = next((s for s in label_dict if 
         s['filename'] == r['query_filename'] 
         and int(s['phrase_id']) == r['query_segment_id']), None)
        algkeys = r['matches'].keys()
        dict_entry = {a: r['matches'][a]['similarity'] for a in algkeys}
        ann_matches_this_melody = [s for s in label_dict if 
         s['filename']==r['match_filename']]
        if not matched_phrase :
            print(r['query_filename'],r['query_segment_id'])
            continue
        for ankey in annotator_keys:
            comparisons = [] 
            for m in ann_matches_this_melody:
                comparisons.append(annotated_phrase_identity(matched_phrase,
                 m,annotator_keys)[ankey])
            best_matching_label_sim = max(comparisons)
            dict_entry[ankey] = best_matching_label_sim
        query_segment_id = "-"+str(r['query_segment_id']) + "-"
        dict_entry['id'] = (r['query_filename'] + query_segment_id + 
         r['match_filename'])
        matches_and_annotations.append(dict_entry)
    return matches_and_annotations


def return_F_score(precision, recall, beta=1.0):
    """ given a precision and recall value, and (optional) beta for the 
    weight of the two measures, returns the F-score. """
    if precision==0.0 or recall==0.0:
        return 0
    else:
        return ((1+beta*beta) * (precision * recall) / 
         (beta*beta* (precision + recall)))