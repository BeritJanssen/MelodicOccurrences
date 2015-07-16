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

from music21 import *
import numpy as np

def adjust_pitches(mel_dict):
    """ takes a dicionary of melodies, calculates the pitch shifts per 
    tune family using pitch histogram intersection and returns the dictionary 
    of melodies after applying pitch shift """
    tunefams = set([m['tunefamily_id'] for m in mel_dict])
    for t in tunefams :
        melodies = [m for m in mel_dict if m['tunefamily_id']==t]
        for i,mel in enumerate(melodies) :
            if i==0:
                continue
            else:
                pitch_shift = get_pitch_shift(melodies[0]['pitch12Histogram'],
                 mel['pitch12Histogram'])
                for s in mel['symbols']:
                    s['pitch'] += pitch_shift
    return mel_dict

def extract_curves_from_corpus(corpus_path, meta_dict=[]):
    """ takes a corpus path, 
    and a dictionary with metadata about the corpus
    returns a dictionary with per melody:
    - tune family id
    - filename
    - pitch histogram of the melody (weighed by duration)
    - per note: 
        - midi note number
        - pitch interval to preceding note
        - onset
        - inter-onset interval
        - ioi ratio with preceding note
        - metric weight of note
        - in which phrase the note occurs
        - phrase position of note
        - scale degree of note
        - note index
    """
    # loop through phrases per song, make dict
    mel_dict = []
    melodies = set([a['filename'] for a in meta_dict])
    for m in melodies:
        pitch_hist = []
        symbols = []
        melody = converter.parse(corpus_path + m + ".krn")
        phrase_onsets = [sp.offset for sp in melody if 
         sp not in melody.parts and sp.offset > 0.0]
        mel = melody.flat
        this_key = mel.getElementsByClass(key.Key)
        if not this_key:
            key_shift = None
        else:
            key_shift = this_key[0].tonic.diatonicNoteNum
        total_duration = mel.duration.quarterLength
        tune = mel.stripTies().notes
        # pitches, pitch intervals, scale degrees
        pitches = [t.midi for t in tune]
        pInt = [pitches[i] - pitches[i-1] 
         for i,p in enumerate(pitches) if i > 0]
        if key_shift:
            sd = [(t.diatonicNoteNum - key_shift)%7 + 1 for t in tune]
        else:
            sd = [None for t in tune]
        # onsets, iois, ioiR
        onsets = [t.offset for t in tune]
        iois = [onsets[i+1] - onsets[i] for i,o in enumerate(onsets) if 
         i < len(onsets) - 1]
        iois.append(tune[-1].quarterLength)
        ioiR = [iois[i]/iois[i-1] for i,o in enumerate(iois) if i > 0]
        # pitch histogram
        set_pitches = set(pitches)
        for s in set_pitches:
            this_pitch = [i for i,p in enumerate(pitches) if p==s]
            hist_weight = sum([iois[t] for t in this_pitch])/total_duration
            pitch_hist.append({"pitch12":s,"value":hist_weight})
        phrases = [a for a in meta_dict if a['filename']==m]
        phrase_num = 0
        for j in range(len(tune)):
            if j==0 :
                symbols.append({'pitch':pitches[j],'pitch_interval':None,
                'onset':onsets[j],'ioi':iois[j],'ioiR':None, 'phrase_id':0,
                'scale_degree':sd[j],
                'note_index':j
                })
            else :
                if onsets[j] >= phrase_onsets[phrase_num]:
                    phrase_num += 1
                symbols.append({'pitch': pitches[j],'pitch_interval': pInt[j-1],
                'onset': onsets[j],'ioi': iois[j],'ioiR': ioiR[j-1],
                'phrase_id': phrase_num,
                'scale_degree': sd[j],
                'note_index': j
                })
        # calculate phrase positions
        phrase_nums = set([s['phrase_id'] for s in symbols])
        for p in phrase_nums :
            phr_subset = [s for s in symbols if s['phrase_id']==p]
            phr_length = len(phr_subset)
            for i,s in enumerate(phr_subset) :
                s['phrasePosition'] = (i+1)/float(phr_length)
        mel_dict.append({'tunefamily_id':phrases[0]['tunefamily_id'], 
            'filename':m, 'pitch12Histogram':pitch_hist,'symbols':symbols})
    return mel_dict
    
def filter_phrases(mel_dict):
    """ this function takes a dictionary of melodies, and returns a dictionary 
    of phrases (according to phrase boundaries in the *kern file)"""
    phrase_dict = []
    for m in mel_dict:
        num_phrases = set([s['phrase_id'] for s in m['symbols']])
        for p in num_phrases :
            selection = [s for s in m['symbols'] if s['phrase_id']==p]
            phrase_dict.append({'tunefamily_id': m['tunefamily_id'],
             'filename': m['filename'], 
             'pitch12Histogram': m['pitch12Histogram'],
             'segment_id': p, 'symbols': selection})
    return phrase_dict

def get_pitch_shift(hist1, hist2):
	""" takes two pitch histograms and determines how much 
    the second melody needs to be shifted wrt the first
	code: Peter van Kranenburg """
	h1 = np.zeros(120)
	h2 = np.zeros(120)
	for i in hist1:
		h1[i['pitch12']] = i['value']
	for i in hist2:
		h2[i['pitch12']] = i['value']
	h1 = np.lib.pad(h1, (120,120), 'constant', constant_values=(0,0))
	max_int = -120
	shift = 0
	for k in range(240):
		intersection = sum(np.minimum(h1[k:k+120],h2))
		if intersection > max_int:
			max_int = intersection
			shift = k
	return shift-120
    
def make_duration_weighted_pitch_sequences(mel_dict,sampling_rate):
    """this function takes a dictionary of melodies or phrases 
    and an indication how often 
    per quarter note a melody is to be sampled (sampling_rate)
    returns duration weighted pitch sequences
    """
    mel_dict_dw = []
    for m in mel_dict:
        pitch_sequence = []
        for s in m['symbols'] :
            repeat = int(round(s['ioi']*sampling_rate))
            pitch_sequence.extend([s['pitch']]*repeat)
        dict_entry = {'filename': m['filename'],
         'tunefamily_id': m['tunefamily_id'],
         'symbols': [{'pitch': p} for p in pitch_sequence]}
        if 'segment_id' in m:
            dict_entry['segment_id'] = m['segment_id']
        mel_dict_dw.append(dict_entry)
    return mel_dict_dw