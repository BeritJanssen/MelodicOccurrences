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

import csv

def add_tunefamily_ids(in_dict,conversion_table_path):
    """ This function takes a list of dictionaries as produced by "csv_to_dict"
    and the path to a csv containing the tune family names and identifiers,
    and adds the identifiers to the in_dict list.
    This is needed for compatibility of ANN2.0 (which has tune family names) 
    with FS1.0 (which has tune family identifiers).
    Most function within MelodicOccurrences sort by tune family identifiers.
    """
    mapping = csv_to_dict(conversion_table_path)
    for entry in in_dict:
        tf = next((m for m in mapping if m['tunefamily']==entry['tunefamily']),None)
        entry['tunefamily_id'] = tf['tunefamily_id']
    return in_dict

def csv_to_dict(doc, keys=None, deli=","):
    """ 
    This function takes a csv file and returns a list of dictionaries.
    Arguments: doc=csv file, keys=the keys assigned to the columns,
    if None, the first row of the csv is converted to the keys 
    deli=delimiter to use (e.g. \t for tab)
    """
    dict_list = []
    with open(doc, "rU") as f:
        read = csv.DictReader(f, keys, delimiter=deli)
        for line in read:
            dict_list.append(line)
        return dict_list

def dict_to_csv(in_dict, keys, fname, deli=","):
    """ this function takes a dictionary and its keys, creates a csv file
    (e.g. for analysis in R or other software)
    """
    with open(fname, "w+") as f:
        wr = csv.writer(f, delimiter=deli)
        wr.writerow(keys)
        for item in in_dict :
            elements = [item[k] for k in keys]
            wr.writerow(elements)

def save_for_R(evaluation_list, fname):
    out_dict = []
    general_info = ('match_filename','query_filename','query_segment_id','tunefamily_id')
    for e in evaluation_list:
        if e['match_filename']==e['query_filename']:
            continue
        position_eval = e['position_eval']
        for p in position_eval:
            for key in general_info:
                p[key] = e[key]
        out_dict.extend(position_eval)
    dict_to_csv(out_dict,list(out_dict[0].keys()), fname)