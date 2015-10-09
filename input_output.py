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